#!/usr/bin/env python3
"""
planted_clique_actual_loss_gradient_lpr.py

Edit CONFIG below. Set SKIP_TRAINING=True to load checkpoints and run eval only.

This version fixes the model-aware variant:

    The aware GNN does NOT receive degree as an explicit input feature.
    Both standard and aware GNNs receive only the alive-mask feature.

    The aware GNN injects the gradient of the actual clique loss internally
    into the message-passing update, similar in spirit to OptGNN / QuerySAT-style
    gradient-aware updates.

At each GNN layer:
    0. aggregate neighbors using a sum-based operator so degree information is not erased,
    1. compute current node scores from current hidden states,
    2. convert scores to probabilities p = sigmoid(scores) * alive,
    3. compute grad_p L(p; A, k), the gradient of the actual clique loss,
    4. concatenate that gradient to the message-passing update.

Models:
  1. standard GNN + objective training
  2. gradient-aware GNN + objective training
  3. standard GNN + oracle incremental-LPR imitation training
  4. gradient-aware GNN + oracle incremental-LPR imitation training
  5. standard GNN + self-predictor LPR training
  6. gradient-aware GNN + self-predictor LPR training

Decoders:
  A. one_pass:
     Run GNN once and rank vertices by score.

  B. rerun_pruned:
     Repeatedly rerun GNN on the residual graph and remove the lowest score.

  C. incremental:
     Run GNN once, then update scores using a learned O(n)-per-removal MLP updater
     without rerunning the GNN.

Metrics:
  approx  = |found valid clique| / |planted clique|
  overlap = |found clique ∩ planted clique| / |planted clique|
  exact   = exact planted clique recovery rate
"""

from __future__ import annotations

import random
import csv
import json
from pathlib import Path
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from scipy import stats


# ============================================================
# CONFIG
# ============================================================

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_N = 1000
TEST_N = 1000
P_ER = 0.5

EPOCHS = 120
BATCH_GRAPHS = 1
HIDDEN = 64
INTERNAL_STEPS = 10  # one shared message-passing layer applied this many times
ACTIVATION = "tanh"  # choose: relu, tanh, sigmoid, gelu, elu, leaky_relu, identity
AGGREGATION = "sum_norm_n"  # choose: "mean", "sum", "sum_norm_n", "sum_norm_sqrt_n"
LR = 1e-3

TRAIN_K_MIN = 10
TRAIN_K_MAX = 100

IMITATION_STEPS = 64
IMITATION_WEIGHT = 1.0
# Self-predictor LPR should run the full residual trajectory until k nodes remain.
# Set to None for full n-k steps. Use an int only for faster debugging.
SELF_LPR_STEPS = 100
SELF_LPR_WEIGHT = 1.0
OBJECTIVE_WEIGHT = 1.0

EVAL_GRAPHS_PER_REGIME = 10
RUN_PC1_RERUN_DECODER = False  # pc1_rerun_pruned is very slow: PCA at every pruning step
BASE_EVAL_DECODERS = [
    # "one_pass", "rerun_pruned",
    "incremental", "pc1_one_pass", "pc1_incremental"
]
EVAL_DECODERS = BASE_EVAL_DECODERS + (["pc1_rerun_pruned"] if RUN_PC1_RERUN_DECODER else [])
PRINT_EVAL_PROGRESS = True
PRINT_PRUNE_EVERY = 100

SAVE_OUTPUTS = True
SKIP_TRAINING = False  # new learned-updater model needs new checkpoints
LEARNED_UPDATER_HIDDEN = 64
RUN_NAME = "residual_gnn_learned_updater"
OUTPUT_DIR = Path("runs") / RUN_NAME
MODEL_DIR = OUTPUT_DIR / "models"
PER_INSTANCE_CSV = OUTPUT_DIR / "per_instance_results.csv"
AGGREGATE_CSV = OUTPUT_DIR / "aggregate_results.csv"
RUN_CONFIG_JSON = OUTPUT_DIR / "run_config.json"

REGIMES = {
    "easy": (62, 100),
    "medium": (36, 61),
    "hard": (20, 35),
}



def ensure_output_dirs() -> None:
    if not SAVE_OUTPUTS:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def save_run_config() -> None:
    if not SAVE_OUTPUTS:
        return

    config = {
        "SEED": SEED,
        "DEVICE": DEVICE,
        "TRAIN_N": TRAIN_N,
        "TEST_N": TEST_N,
        "P_ER": P_ER,
        "EPOCHS": EPOCHS,
        "BATCH_GRAPHS": BATCH_GRAPHS,
        "HIDDEN": HIDDEN,
        "INTERNAL_STEPS": INTERNAL_STEPS,
        "ACTIVATION": ACTIVATION,
        "AGGREGATION": AGGREGATION,
        "LR": LR,
        "TRAIN_K_MIN": TRAIN_K_MIN,
        "TRAIN_K_MAX": TRAIN_K_MAX,
        "IMITATION_STEPS": IMITATION_STEPS,
        "SELF_LPR_STEPS": SELF_LPR_STEPS,
        "OBJECTIVE_WEIGHT": OBJECTIVE_WEIGHT,
        "IMITATION_WEIGHT": IMITATION_WEIGHT,
        "SELF_LPR_WEIGHT": SELF_LPR_WEIGHT,
        "EVAL_GRAPHS_PER_REGIME": EVAL_GRAPHS_PER_REGIME,
        "RUN_PC1_RERUN_DECODER": RUN_PC1_RERUN_DECODER,
        "EVAL_DECODERS": EVAL_DECODERS,
        "REGIMES": REGIMES,
        "SKIP_TRAINING": SKIP_TRAINING,
        "LEARNED_UPDATER_HIDDEN": LEARNED_UPDATER_HIDDEN,
        "RUN_NAME": RUN_NAME,
    }

    with open(RUN_CONFIG_JSON, "w") as f:
        json.dump(config, f, indent=2)


def save_model_checkpoint(model: nn.Module, spec: "TrainSpec") -> None:
    if not SAVE_OUTPUTS:
        return

    path = MODEL_DIR / f"{spec.name}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "spec": asdict(spec),
            "hidden": HIDDEN,
            "internal_steps": INTERNAL_STEPS,
            "activation": ACTIVATION,
            "aggregation": AGGREGATION,
            "gradient_aware": spec.gradient_aware,
            "learned_updater_hidden": LEARNED_UPDATER_HIDDEN,
        },
        path,
    )


def load_model_checkpoint(spec: "TrainSpec", device: torch.device) -> nn.Module:
    path = MODEL_DIR / f"{spec.name}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {path}. Train first (SKIP_TRAINING=False) or set RUN_NAME to match saved models."
        )

    checkpoint = torch.load(path, map_location=device)
    model = SimpleGNN(
        hidden=checkpoint.get("hidden", HIDDEN),
        layers=checkpoint.get("internal_steps", INTERNAL_STEPS),
        activation=checkpoint.get("activation", ACTIVATION),
        gradient_aware=checkpoint.get("gradient_aware", spec.gradient_aware),
        updater_hidden=checkpoint.get("learned_updater_hidden", LEARNED_UPDATER_HIDDEN),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint: {path}", flush=True)
    return model


def write_csv(path: Path, rows: list[dict]) -> None:
    if not SAVE_OUTPUTS or not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}

    for row in rows:
        key = (
            row["model"],
            row["gradient_aware"],
            row["training_way"],
            row["regime"],
            row["decoder"],
        )
        groups.setdefault(key, []).append(row)

    agg_rows: list[dict] = []
    for (model, gradient_aware, training_way, regime, decoder), group in groups.items():
        n_items = len(group)
        agg_rows.append(
            {
                "model": model,
                "gradient_aware": gradient_aware,
                "training_way": training_way,
                "regime": regime,
                "decoder": decoder,
                "num_instances": n_items,
                "approx_mean": sum(r["approx"] for r in group) / n_items,
                "overlap_mean": sum(r["overlap"] for r in group) / n_items,
                "exact_mean": sum(r["exact"] for r in group) / n_items,
            }
        )

    return agg_rows


# ============================================================
# Data
# ============================================================

def make_planted_clique(n: int, k: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    A = torch.bernoulli(torch.full((n, n), P_ER, device=device))
    A = torch.triu(A, diagonal=1)
    A = A + A.t()

    perm = torch.randperm(n, device=device)
    clique = perm[:k]
    A[clique[:, None], clique[None, :]] = 1.0
    A.fill_diagonal_(0.0)

    planted = torch.zeros(n, dtype=torch.bool, device=device)
    planted[clique] = True
    return A, planted


def sample_train_graph(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
    k = random.randint(TRAIN_K_MIN, TRAIN_K_MAX)
    A, planted = make_planted_clique(TRAIN_N, k, device)
    return A, planted, k


def make_features(A: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
    """
    Both standard and gradient-aware models receive the same explicit input:
    only the alive-mask feature.

    No degree feature is given.
    """
    return alive.float().unsqueeze(-1)


# ============================================================
# Clique metrics
# ============================================================

def is_clique(A: torch.Tensor, selected: torch.Tensor) -> bool:
    idx = selected.nonzero(as_tuple=False).flatten()
    k = int(idx.numel())
    if k <= 1:
        return True
    sub = A[idx[:, None], idx[None, :]]
    return float(sub.sum().item()) == k * (k - 1)


def greedy_clique_from_order(A: torch.Tensor, order: torch.Tensor, max_k: int | None = None) -> torch.Tensor:
    n = A.shape[0]
    selected = torch.zeros(n, dtype=torch.bool, device=A.device)
    chosen: list[int] = []

    for v_t in order:
        v = int(v_t.item())
        if max_k is not None and len(chosen) >= max_k:
            break

        if not chosen:
            selected[v] = True
            chosen.append(v)
            continue

        prev = torch.tensor(chosen, dtype=torch.long, device=A.device)
        if bool((A[v, prev] > 0.5).all().item()):
            selected[v] = True
            chosen.append(v)

    return selected


def is_alive_subgraph_clique(A: torch.Tensor, alive: torch.Tensor) -> bool:
    """True iff the currently alive vertices induce a clique."""
    idx = alive.nonzero(as_tuple=False).flatten()
    m = int(idx.numel())
    if m <= 1:
        return True
    sub = A[idx[:, None], idx[None, :]]
    return float(sub.sum().item()) == m * (m - 1)


def can_add_to_clique(A: torch.Tensor, clique: torch.Tensor, v: torch.Tensor) -> bool:
    """True iff v connects to all vertices currently in clique."""
    idx = clique.nonzero(as_tuple=False).flatten()
    if idx.numel() == 0:
        return True
    return bool((A[v, idx] > 0.5).all().item())


def add_back_removed_nodes(A: torch.Tensor, alive_clique: torch.Tensor, removed: list[torch.Tensor]) -> torch.Tensor:
    """
    LDR-style add-back phase.

    After pruning until the alive set is a clique, scan removed vertices in
    reverse removal order. Add a vertex back iff it keeps the set a clique,
    i.e. iff it increases the clique size by one.
    """
    clique = alive_clique.clone()

    for v in reversed(removed):
        if can_add_to_clique(A, clique, v):
            clique[v] = True

    return clique


def init_alive_degrees_and_edges(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize residual degrees and number of undirected alive edges.

    deg_alive[i] = number of alive neighbors of i.
    alive_edges = number of undirected edges inside the alive set.
    """
    deg_alive = A.sum(dim=1)
    alive_edges = A.sum() / 2.0
    return deg_alive, alive_edges


def alive_is_clique_fast(alive_count: int, alive_edges: torch.Tensor) -> bool:
    """
    O(1) clique check.

    A graph on m alive vertices is a clique iff it has m(m-1)/2 undirected edges.
    """
    if alive_count <= 1:
        return True
    target_edges = alive_count * (alive_count - 1) / 2.0
    return bool(alive_edges.item() == target_edges)


def remove_vertex_update_state(
    A: torch.Tensor,
    alive: torch.Tensor,
    deg_alive: torch.Tensor,
    alive_edges: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Remove v and update alive mask, residual degrees, and alive edge count.

    Complexity: O(n), no submatrix construction.
    """
    v_int = int(v.item())

    alive_next = alive.clone()
    alive_next[v_int] = False

    # v contributes deg_alive[v] undirected edges to the alive induced subgraph.
    alive_edges_next = alive_edges - deg_alive[v_int]

    # Every alive neighbor of v loses one residual degree.
    deg_alive_next = deg_alive - A[:, v_int]
    deg_alive_next[v_int] = 0.0
    deg_alive_next = deg_alive_next * alive_next.float()

    return alive_next, deg_alive_next, alive_edges_next




def approx_ratio(found_clique: torch.Tensor, planted: torch.Tensor) -> float:
    k = int(planted.sum().item())
    return 0.0 if k == 0 else float(found_clique.sum().item()) / k


def planted_overlap(found_clique: torch.Tensor, planted: torch.Tensor) -> float:
    k = int(planted.sum().item())
    return 0.0 if k == 0 else float((found_clique & planted).sum().item()) / k


# ============================================================
# Actual loss and its gradient wrt p
# ============================================================

def clique_loss_from_probs(p: torch.Tensor, A: torch.Tensor, alive: torch.Tensor, k: int) -> torch.Tensor:
    """
    Same unsupervised clique loss, evaluated on the residual/alive graph.

    Dead nodes do not participate in edge/nonedge interactions.
    """
    n = A.shape[0]
    alive_vec = alive.float()
    p = p * alive_vec

    A_res = A * alive_vec[:, None] * alive_vec[None, :]

    nonedge = 1.0 - A
    nonedge = nonedge.clone()
    nonedge.fill_diagonal_(0.0)
    nonedge_res = nonedge * alive_vec[:, None] * alive_vec[None, :]

    edge_reward = (p @ A_res @ p) / max(k * (k - 1), 1)
    nonedge_penalty = (p @ nonedge_res @ p) / max(k * (n - k), 1)
    size_penalty = ((p.sum() - k) / n).pow(2)

    return -edge_reward + nonedge_penalty + 10.0 * size_penalty


def clique_loss_gradient_wrt_p(p: torch.Tensor, A: torch.Tensor, alive: torch.Tensor, k: int) -> torch.Tensor:
    """
    Analytic gradient of clique_loss_from_probs wrt p on the residual graph.

    This is the model-aware signal.

    It is NOT a hand-coded degree feature.
    It is the gradient of the actual loss used for training/evaluation.

    Critical residual fix:
      after pruning, dead nodes must not contribute to A @ p or nonedge @ p.
    """
    n = A.shape[0]
    alive_vec = alive.float()
    p = p * alive_vec

    # Residual adjacency: only alive-alive interactions remain.
    A_res = A * alive_vec[:, None] * alive_vec[None, :]

    nonedge = (1.0 - A)
    nonedge = nonedge.clone()
    nonedge.fill_diagonal_(0.0)
    nonedge_res = nonedge * alive_vec[:, None] * alive_vec[None, :]

    # Keep the same k-normalization as the objective, but compute interactions
    # on the residual graph.
    edge_den = max(k * (k - 1), 1)
    nonedge_den = max(k * (n - k), 1)

    grad = (
        -2.0 * (A_res @ p) / edge_den
        + 2.0 * (nonedge_res @ p) / nonedge_den
        + 20.0 * (p.sum() - k) / (n * n)
    )

    grad = grad * alive_vec

    alive_grad = grad[alive.bool()]
    if alive_grad.numel() > 1:
        grad = grad - alive_grad.mean()
        grad = grad / alive_grad.std().clamp_min(1e-6)

    return grad


def clique_objective_loss(scores: torch.Tensor, A: torch.Tensor, alive: torch.Tensor, k: int) -> torch.Tensor:
    p = torch.sigmoid(scores) * alive.float()
    return clique_loss_from_probs(p, A, alive, k)


# ============================================================
# GNN
# ============================================================

def make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1)
    if name == "identity":
        return nn.Identity()
    raise ValueError(name)


class SimpleGNN(nn.Module):
    """
    Standard model:
        h_i^{t+1} = MLP([h_i^t, mean_neighbor(h)^t])

    Gradient-aware model:
        h_i^{t+1} = MLP([h_i^t, mean_neighbor(h)^t, grad_i^t])

    where grad_i^t is grad_p L(p^t; A, k)_i and
    p^t = sigmoid(score_i^t) * alive_i.
    """

    def __init__(
        self,
        hidden: int,
        layers: int,
        activation: str,
        gradient_aware: bool,
        updater_hidden: int = LEARNED_UPDATER_HIDDEN,
    ):
        super().__init__()
        self.gradient_aware = gradient_aware

        self.input = nn.Linear(1, hidden)
        self.out = nn.Linear(hidden, 1)

        self.internal_steps = layers
        update_in_dim = 2 * hidden + (1 if gradient_aware else 0)
        self.update = nn.Sequential(
            nn.Linear(update_in_dim, hidden),
            make_activation(activation),
            nn.Linear(hidden, hidden),
            make_activation(activation),
        )

        # Learned incremental score updater.
        #
        # Input per alive node i:
        #   [s_i, s_v, A_iv, grad_i_or_0, grad_v_or_0, alive_i]
        #
        # Output:
        #   delta_i, applied as s_i <- s_i + delta_i.
        #
        # This replaces the wrong fixed rule s_i <- s_i - A_iv.
        self.score_updater = nn.Sequential(
            nn.Linear(6, updater_hidden),
            make_activation(activation),
            nn.Linear(updater_hidden, updater_hidden),
            make_activation(activation),
            nn.Linear(updater_hidden, 1),
        )

    def _aggregate(self, A: torch.Tensor, h: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
        """
        Residual/alive-aware message aggregation.

        Critical fix:
          removed nodes must not send messages. Merely setting their input
          feature to zero is not enough, because Linear layers have bias and
          can create nonzero hidden states for dead nodes.

        Therefore:
          1. zero hidden states of dead nodes;
          2. use only columns of A corresponding to alive senders.

        This makes rerun-pruned genuinely equivalent to running the GNN on the
        pruned/residual graph.
        """
        n = A.shape[0]
        alive_f = alive.float().unsqueeze(-1)

        # Dead nodes send no messages.
        h_alive = h * alive_f

        # A_res[i,j] is active only if sender j is alive.
        A_res = A * alive.float()[None, :]

        if AGGREGATION == "mean":
            deg = A_res.sum(dim=1, keepdim=True).clamp_min(1.0)
            return (A_res @ h_alive) / deg

        if AGGREGATION == "sum":
            return A_res @ h_alive

        if AGGREGATION == "sum_norm_n":
            return (A_res @ h_alive) / max(n - 1, 1)

        if AGGREGATION == "sum_norm_sqrt_n":
            return (A_res @ h_alive) / (max(n - 1, 1) ** 0.5)

        raise ValueError(f"Unknown AGGREGATION={AGGREGATION}")

    def _hidden(self, A: torch.Tensor, x: torch.Tensor, alive: torch.Tensor, k: int) -> torch.Tensor:
        alive_f = alive.float().unsqueeze(-1)

        # Critical fix: zero dead-node embeddings immediately after input
        # projection, because Linear bias can otherwise make dead nodes nonzero.
        h = self.input(x) * alive_f

        for _ in range(self.internal_steps):
            msg = self._aggregate(A, h, alive)
            parts = [h, msg]

            if self.gradient_aware:
                current_scores = self.out(h).squeeze(-1)
                p = torch.sigmoid(current_scores) * alive.float()
                grad = clique_loss_gradient_wrt_p(p, A, alive, k).unsqueeze(-1)
                parts.append(grad)

            # Critical fix: zero dead-node embeddings after every shared update.
            h = self.update(torch.cat(parts, dim=-1)) * alive_f

        return h

    def forward(self, A: torch.Tensor, x: torch.Tensor, alive: torch.Tensor, k: int) -> torch.Tensor:
        h = self._hidden(A, x, alive, k)
        scores = self.out(h).squeeze(-1)
        return scores.masked_fill(~alive.bool(), -1e9)

    def node_embeddings(self, A: torch.Tensor, x: torch.Tensor, alive: torch.Tensor, k: int) -> torch.Tensor:
        return self._hidden(A, x, alive, k)

    def update_scores(
        self,
        scores: torch.Tensor,
        A: torch.Tensor,
        alive: torch.Tensor,
        removed: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """
        Learned O(n)-per-removal incremental score update.

        This is the correct replacement for the fixed degree-style rule.
        The updater is trained by the incremental imitation / self-predictor
        regimes. It does not receive explicit degree as input.

        It can receive a gradient channel only when the model is gradient-aware;
        otherwise the gradient channel is zero.
        """
        n = A.shape[0]
        alive_f = alive.float()

        removed_score = scores[removed].expand(n)
        affected = A[:, int(removed.item())]

        if self.gradient_aware:
            p = torch.sigmoid(scores) * alive_f
            grad = clique_loss_gradient_wrt_p(p, A, alive, k)
            removed_grad = grad[removed].expand(n)
        else:
            grad = torch.zeros_like(scores)
            removed_grad = torch.zeros_like(scores)

        updater_x = torch.stack(
            [
                scores,
                removed_score,
                affected,
                grad,
                removed_grad,
                alive_f,
            ],
            dim=1,
        )

        delta = self.score_updater(updater_x).squeeze(-1)
        updated = scores + delta

        # Dead nodes must never be selected.
        return updated.masked_fill(~alive.bool(), -1e9)


# ============================================================
# LPR update consistency
# ============================================================

def lpr_remove_lowest(scores: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
    return scores.masked_fill(~alive.bool(), 1e9).argmin()


def oracle_lpr_remove_lowest_degree(A: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
    """
    Teacher-forced LPR oracle: remove the alive node with the lowest residual degree.

    Degree is used only to define the oracle action, not as model input and not
    as a regression target.
    """
    d = (A * alive.float()[None, :]).sum(dim=1)
    return d.masked_fill(~alive.bool(), 1e9).argmin()


def update_scores_lpr(scores: torch.Tensor, A: torch.Tensor, removed: torch.Tensor) -> torch.Tensor:
    """
    Deprecated fixed update. Kept only to catch accidental uses.

    Incremental decoding should use:
        model.update_scores(scores, A, alive, removed, k)

    The fixed update s_i <- s_i - A[i,v] is only valid when scores are already
    degree-like. That is not assumed here.
    """
    raise RuntimeError("Do not use fixed update_scores_lpr; use model.update_scores(...) instead.")


def incremental_lpr_imitation_loss(model: nn.Module, A: torch.Tensor, k: int, steps: int) -> torch.Tensor:
    """
    Teacher-forced training for the incremental decoder.

    This is option 1: mimic LPR/LDR behavior directly.

    Procedure:
      1. Run the GNN once on the full graph.
      2. For several residual steps:
            target = oracle node removed by LPR, i.e. lowest residual-degree node.
            train current scores to remove that target.
            teacher-force the removal of target.
            update scores by the same cheap rule used at test time:
                s_i <- s_i - A[i, target]

    Important:
      - degree is not an explicit model input.
      - degree is not used as an MSE/regression target.
      - degree only defines the oracle discrete action.
      - the model is trained under the same incremental update dynamics used at test time.
    """
    n = A.shape[0]
    alive = torch.ones(n, dtype=torch.bool, device=A.device)

    x = make_features(A, alive)
    scores = model(A, x, alive, k)

    losses = []
    max_steps = (n - k) if steps is None else min(steps, n - k)

    for _ in range(max_steps):
        target = oracle_lpr_remove_lowest_degree(A, alive)

        # Cross entropy selects the node to REMOVE.
        # Since lower score means remove, use -scores as removal logits.
        removal_logits = (-scores).masked_fill(~alive.bool(), -1e9)
        losses.append(F.cross_entropy(removal_logits.unsqueeze(0), target.unsqueeze(0)))

        # Teacher-forced residual transition.
        # Avoid in-place mutation of alive because it participates in the
        # graph-tracked score computation.
        alive_next = alive.clone()
        alive_next[target] = False
        alive = alive_next

        scores = model.update_scores(
            scores=scores,
            A=A,
            alive=alive,
            removed=target,
            k=k,
        )
        scores = scores.masked_fill(~alive.bool(), -1e9)

        if int(alive.sum().item()) <= k:
            break

    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=A.device)



def self_predictor_lpr_loss(model: nn.Module, A: torch.Tensor, k: int, steps: int | None) -> torch.Tensor:
    """
    Self-predictor LPR training.

    Goal:
      Train the one-pass incremental scores to mimic the model's own
      rerun-pruned behavior.

    This does NOT use degree:
      - no degree input,
      - no degree regression target,
      - no degree oracle action.

    Procedure:
      Maintain two residual trajectories:

      1. Teacher trajectory:
         rerun the model on the current residual graph and remove the model's
         own lowest-scoring node.

      2. Student trajectory:
         run the model once at the beginning, then update scores cheaply with
         s_i <- s_i - A[i, removed].

      The loss asks the student incremental scores to remove the same node as
      the teacher rerun model at each step.

    Interpretation:
      This trains the incremental decoder to approximate the expensive
      rerun-pruned decoder, using the model's own predictor as the teacher.

      With SELF_LPR_STEPS = None, this runs the full residual trajectory until
      only k nodes remain, i.e. n-k removal steps.
    """
    n = A.shape[0]

    # Student: one initial GNN pass, then cheap score updates only.
    alive_student = torch.ones(n, dtype=torch.bool, device=A.device)
    student_scores = model(A, make_features(A, alive_student), alive_student, k)

    # Teacher: rerun GNN on every residual graph.
    alive_teacher = torch.ones(n, dtype=torch.bool, device=A.device)

    losses = []
    max_steps = (n - k) if steps is None else min(steps, n - k)

    for _ in range(max_steps):
        # Teacher action: model's own rerun-pruned prediction.
        with torch.no_grad():
            teacher_scores = model(A, make_features(A, alive_teacher), alive_teacher, k)
            target = lpr_remove_lowest(teacher_scores, alive_teacher)

        # Student action should match teacher action under current incremental scores.
        removal_logits = (-student_scores).masked_fill(~alive_student.bool(), -1e9)
        losses.append(F.cross_entropy(removal_logits.unsqueeze(0), target.unsqueeze(0)))

        # Teacher and student follow the same teacher-forced removal.
        # Do NOT modify alive masks in-place. They are used inside graph-tracked
        # operations, and in-place edits break autograd during the full rollout.
        alive_teacher_next = alive_teacher.clone()
        alive_student_next = alive_student.clone()
        alive_teacher_next[target] = False
        alive_student_next[target] = False

        alive_teacher = alive_teacher_next
        alive_student = alive_student_next

        # Student uses the learned incremental updater.
        student_scores = model.update_scores(
            scores=student_scores,
            A=A,
            alive=alive_student,
            removed=target,
            k=k,
        )
        student_scores = student_scores.masked_fill(~alive_student.bool(), -1e9)

        if int(alive_student.sum().item()) <= k:
            break

    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=A.device)


def train_loss(model: nn.Module, A: torch.Tensor, k: int, training_way: str) -> torch.Tensor:
    alive = torch.ones(A.shape[0], dtype=torch.bool, device=A.device)
    scores = model(A, make_features(A, alive), alive, k)

    obj = clique_objective_loss(scores, A, alive, k)

    if training_way == "objective":
        return obj

    if training_way == "incremental_lpr_imitation":
        imitation = incremental_lpr_imitation_loss(model, A, k, IMITATION_STEPS)
        return OBJECTIVE_WEIGHT * obj + IMITATION_WEIGHT * imitation

    if training_way == "self_predictor_lpr":
        self_lpr = self_predictor_lpr_loss(model, A, k, SELF_LPR_STEPS)
        return OBJECTIVE_WEIGHT * obj + SELF_LPR_WEIGHT * self_lpr

    raise ValueError(training_way)



# ============================================================
# PC1 score extraction
# ============================================================

@torch.no_grad()
def pc1_scores_from_embeddings(model: nn.Module, A: torch.Tensor, alive: torch.Tensor, k: int) -> torch.Tensor:
    """
    Run one GNN pass, take final node embeddings, compute PC1 over alive nodes,
    and use the PC1 coordinate as a linear score.

    Sign is aligned with the model's raw scores on alive nodes so that higher
    PC1 means more likely to keep, matching the normal decoder convention.
    """
    x = make_features(A, alive)
    h = model.node_embeddings(A, x, alive, k)

    alive_idx = alive.bool()
    h_alive = h[alive_idx].detach().cpu().numpy()

    if h_alive.shape[0] <= 1:
        return torch.zeros(A.shape[0], device=A.device).masked_fill(~alive.bool(), -1e9)

    pc1_alive_np = PCA(n_components=1).fit_transform(h_alive).reshape(-1)
    pc1_alive = torch.tensor(pc1_alive_np, dtype=h.dtype, device=A.device)

    # Align sign with raw model score. If correlation is negative, flip PC1.
    raw_scores = model(A, x, alive, k)[alive_idx].detach()
    pc1_centered = pc1_alive - pc1_alive.mean()
    raw_centered = raw_scores - raw_scores.mean()
    sign = torch.sign((pc1_centered * raw_centered).sum())
    if sign.item() == 0:
        sign = torch.tensor(1.0, device=A.device, dtype=h.dtype)

    pc1_alive = pc1_alive * sign

    scores = torch.full((A.shape[0],), -1e9, dtype=h.dtype, device=A.device)
    scores[alive_idx] = pc1_alive
    return scores


@torch.no_grad()
def order_one_pass_pc1(model: nn.Module, A: torch.Tensor, k: int) -> torch.Tensor:
    n = A.shape[0]
    alive = torch.ones(n, dtype=torch.bool, device=A.device)
    scores = pc1_scores_from_embeddings(model, A, alive, k)
    return scores.argsort(descending=True)


@torch.no_grad()
def select_rerun_pruned_pc1_ldr(model: nn.Module, A: torch.Tensor, k: int) -> torch.Tensor:
    """
    Fast PC1 rerun-pruned LDR. Still expensive because PCA is recomputed each
    pruning step, but clique checking is now O(1).
    """
    n = A.shape[0]
    alive = torch.ones(n, dtype=torch.bool, device=A.device)
    deg_alive, alive_edges = init_alive_degrees_and_edges(A)
    alive_count = n
    removed: list[torch.Tensor] = []

    step = 0
    while not alive_is_clique_fast(alive_count, alive_edges):
        # if PRINT_EVAL_PROGRESS and step % PRINT_PRUNE_EVERY == 0:
        #     print(f"      pc1_rerun pruning step={step}, alive={alive_count}", flush=True)

        scores = pc1_scores_from_embeddings(model, A, alive, k)
        v = lpr_remove_lowest(scores, alive)
        removed.append(v)

        alive, deg_alive, alive_edges = remove_vertex_update_state(
            A, alive, deg_alive, alive_edges, v
        )
        alive_count -= 1
        step += 1

    # if PRINT_EVAL_PROGRESS:
    #     print(f"      pc1_rerun stopped: clique alive={alive_count}, add_back={len(removed)}", flush=True)

    return add_back_removed_nodes(A, alive, removed)


@torch.no_grad()
def select_incremental_pc1_ldr(model: nn.Module, A: torch.Tensor, k: int) -> torch.Tensor:
    """
    Fast PC1 incremental LDR: one PC1 extraction, cheap score updates, O(1)
    clique checks.
    """
    n = A.shape[0]
    alive = torch.ones(n, dtype=torch.bool, device=A.device)
    deg_alive, alive_edges = init_alive_degrees_and_edges(A)
    alive_count = n

    scores = pc1_scores_from_embeddings(model, A, alive, k)
    removed: list[torch.Tensor] = []

    step = 0
    while not alive_is_clique_fast(alive_count, alive_edges):
        # if PRINT_EVAL_PROGRESS and step % PRINT_PRUNE_EVERY == 0:
        #     print(f"      incremental pruning step={step}, alive={alive_count}", flush=True)

        v = lpr_remove_lowest(scores, alive)
        removed.append(v)

        alive, deg_alive, alive_edges = remove_vertex_update_state(
            A, alive, deg_alive, alive_edges, v
        )
        alive_count -= 1

        scores = model.update_scores(
            scores=scores,
            A=A,
            alive=alive,
            removed=v,
            k=k,
        )
        scores = scores.masked_fill(~alive.bool(), -1e9)
        step += 1

    # if PRINT_EVAL_PROGRESS:
    #     print(f"      incremental stopped: clique alive={alive_count}, add_back={len(removed)}", flush=True)

    return add_back_removed_nodes(A, alive, removed)


# ============================================================
# Decoders
# ============================================================

@torch.no_grad()
def order_one_pass(model: nn.Module, A: torch.Tensor, k: int) -> torch.Tensor:
    n = A.shape[0]
    alive = torch.ones(n, dtype=torch.bool, device=A.device)
    scores = model(A, make_features(A, alive), alive, k)
    return scores.argsort(descending=True)


@torch.no_grad()
def select_rerun_pruned_ldr(model: nn.Module, A: torch.Tensor, k: int) -> torch.Tensor:
    """
    Fast LDR-style rerun-pruned decoder:
      1. maintain residual edge count for O(1) clique checks;
      2. rerun GNN only for scoring;
      3. stop as soon as alive set is a clique;
      4. add removed nodes back if they preserve the clique.
    """
    n = A.shape[0]
    alive = torch.ones(n, dtype=torch.bool, device=A.device)
    deg_alive, alive_edges = init_alive_degrees_and_edges(A)
    alive_count = n
    removed: list[torch.Tensor] = []

    step = 0
    while not alive_is_clique_fast(alive_count, alive_edges):
        # if PRINT_EVAL_PROGRESS and step % PRINT_PRUNE_EVERY == 0:
        #     print(f"      rerun_pruned pruning step={step}, alive={alive_count}", flush=True)

        scores = model(A, make_features(A, alive), alive, k)
        v = lpr_remove_lowest(scores, alive)
        removed.append(v)

        alive, deg_alive, alive_edges = remove_vertex_update_state(
            A, alive, deg_alive, alive_edges, v
        )
        alive_count -= 1
        step += 1

    # if PRINT_EVAL_PROGRESS:
    #     print(f"      rerun_pruned stopped: clique alive={alive_count}, add_back={len(removed)}", flush=True)

    return add_back_removed_nodes(A, alive, removed)


@torch.no_grad()
def select_incremental_ldr(model: nn.Module, A: torch.Tensor, k: int) -> torch.Tensor:
    """
    Fast LDR-style incremental decoder:
      one GNN pass, cheap score updates, O(1) clique checks.
    """
    n = A.shape[0]
    alive = torch.ones(n, dtype=torch.bool, device=A.device)
    deg_alive, alive_edges = init_alive_degrees_and_edges(A)
    alive_count = n

    scores = model(A, make_features(A, alive), alive, k)
    removed: list[torch.Tensor] = []

    step = 0
    while not alive_is_clique_fast(alive_count, alive_edges):
        # if PRINT_EVAL_PROGRESS and step % PRINT_PRUNE_EVERY == 0:
        #     print(f"      pc1_incremental pruning step={step}, alive={alive_count}", flush=True)

        v = lpr_remove_lowest(scores, alive)
        removed.append(v)

        alive, deg_alive, alive_edges = remove_vertex_update_state(
            A, alive, deg_alive, alive_edges, v
        )
        alive_count -= 1

        scores = model.update_scores(
            scores=scores,
            A=A,
            alive=alive,
            removed=v,
            k=k,
        )
        scores = scores.masked_fill(~alive.bool(), -1e9)
        step += 1

    # if PRINT_EVAL_PROGRESS:
    #     print(f"      pc1_incremental stopped: clique alive={alive_count}, add_back={len(removed)}", flush=True)

    return add_back_removed_nodes(A, alive, removed)


@torch.no_grad()
def eval_one_decoder(model: nn.Module, A: torch.Tensor, planted: torch.Tensor, k: int, decoder: str) -> tuple[float, float, bool]:
    if decoder == "one_pass":
        order = order_one_pass(model, A, k)
        found = greedy_clique_from_order(A, order, max_k=k)
    elif decoder == "rerun_pruned":
        found = select_rerun_pruned_ldr(model, A, k)
    elif decoder == "incremental":
        found = select_incremental_ldr(model, A, k)
    elif decoder == "pc1_one_pass":
        order = order_one_pass_pc1(model, A, k)
        found = greedy_clique_from_order(A, order, max_k=k)
    elif decoder == "pc1_rerun_pruned":
        found = select_rerun_pruned_pc1_ldr(model, A, k)
    elif decoder == "pc1_incremental":
        found = select_incremental_pc1_ldr(model, A, k)
    else:
        raise ValueError(decoder)

    approx = approx_ratio(found, planted)
    overlap = planted_overlap(found, planted)
    exact = int(found.sum().item()) >= k and is_clique(A, found)
    return approx, overlap, exact



@torch.no_grad()
def sanity_check_residual_masking(device: torch.device) -> None:
    """
    Quick check: removing nodes must change a standard GNN's scores because
    removed nodes no longer send messages.
    """
    A, _planted = make_planted_clique(80, 20, device)
    model = SimpleGNN(
        hidden=HIDDEN,
        layers=INTERNAL_STEPS,
        activation=ACTIVATION,
        gradient_aware=False,
        updater_hidden=LEARNED_UPDATER_HIDDEN,
    ).to(device)
    alive = torch.ones(80, dtype=torch.bool, device=device)
    s0 = model(A, make_features(A, alive), alive, 20)

    alive2 = alive.clone()
    alive2[:20] = False
    s1 = model(A, make_features(A, alive2), alive2, 20)

    diff = (s0[alive2] - s1[alive2]).abs().mean().item()
    print(f"[sanity] residual masking score diff after pruning = {diff:.6f}", flush=True)


# ============================================================
# Train / evaluate
# ============================================================

@dataclass
class TrainSpec:
    name: str
    gradient_aware: bool
    training_way: str


def train_model(spec: TrainSpec, device: torch.device) -> nn.Module:
    model = SimpleGNN(
        hidden=HIDDEN,
        layers=INTERNAL_STEPS,
        activation=ACTIVATION,
        gradient_aware=spec.gradient_aware,
        updater_hidden=LEARNED_UPDATER_HIDDEN,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    print(f"\nTraining {spec.name}")
    print("=" * (9 + len(spec.name)))

    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()

        losses = []
        for _ in range(BATCH_GRAPHS):
            A, _planted, k = sample_train_graph(device)
            losses.append(train_loss(model, A, k, spec.training_way))

        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()

        if epoch == 1 or epoch % 20 == 0:
            print(f"epoch={epoch:04d} loss={loss.item():.4f}")

    return model


@torch.no_grad()
def evaluate_model(model: nn.Module, spec: TrainSpec, device: torch.device) -> tuple[dict, list[dict]]:
    model.eval()

    decoders = EVAL_DECODERS

    out = {}
    rows: list[dict] = []

    for regime, (k_min, k_max) in REGIMES.items():
        out[regime] = {}

        for decoder in decoders:
            approx_sum = 0.0
            overlap_sum = 0.0
            exact_sum = 0.0

            for graph_idx in range(EVAL_GRAPHS_PER_REGIME):
                k = random.randint(k_min, k_max)
                instance_seed = random.randint(0, 2**31 - 1)

                # Make the generated instance reproducible.
                py_state = random.getstate()
                torch_state = torch.random.get_rng_state()
                cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

                random.seed(instance_seed)
                torch.manual_seed(instance_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(instance_seed)

                if PRINT_EVAL_PROGRESS:
                    print(
                        f"[eval] model={spec.name} regime={regime} decoder={decoder} "
                        f"graph={graph_idx + 1}/{EVAL_GRAPHS_PER_REGIME} k={k} seed={instance_seed}",
                        flush=True,
                    )

                A, planted = make_planted_clique(TEST_N, k, device)

                approx, overlap, exact = eval_one_decoder(
                    model=model,
                    A=A,
                    planted=planted,
                    k=k,
                    decoder=decoder,
                )

                # Restore RNG so the rest of the evaluation remains deterministic.
                random.setstate(py_state)
                torch.random.set_rng_state(torch_state)
                if torch.cuda.is_available() and cuda_state is not None:
                    torch.cuda.set_rng_state_all(cuda_state)

                exact_float = float(exact)

                if PRINT_EVAL_PROGRESS:
                    print(
                        f"[eval done] model={spec.name} regime={regime} decoder={decoder} "
                        f"graph={graph_idx + 1} approx={approx:.3f} "
                        f"overlap={overlap:.3f} exact={exact_float:.3f}",
                        flush=True,
                    )

                row = {
                    "model": spec.name,
                    "gradient_aware": spec.gradient_aware,
                    "training_way": spec.training_way,
                    "regime": regime,
                    "decoder": decoder,
                    "graph_idx": graph_idx,
                    "instance_seed": instance_seed,
                    "n": TEST_N,
                    "k": k,
                    "approx": approx,
                    "overlap": overlap,
                    "exact": exact_float,
                }
                rows.append(row)

                approx_sum += approx
                overlap_sum += overlap
                exact_sum += exact_float

            out[regime][decoder] = {
                "approx": approx_sum / EVAL_GRAPHS_PER_REGIME,
                "overlap": overlap_sum / EVAL_GRAPHS_PER_REGIME,
                "exact": exact_sum / EVAL_GRAPHS_PER_REGIME,
            }

    return out, rows


@torch.no_grad()
def pca_degree_diagnostics_one_graph(model: nn.Module, A: torch.Tensor, k: int) -> dict[str, float]:
    """
    One full-graph GNN pass, PCA on final node embeddings, then:
      - PC1 explained variance
      - PC2 explained variance
      - Spearman(degree, PC1)

    Degree is original row-sum(A). It is not given as an input feature.
    """
    n = A.shape[0]
    alive = torch.ones(n, dtype=torch.bool, device=A.device)
    x = make_features(A, alive)

    h = model.node_embeddings(A, x, alive, k).detach().cpu().numpy()
    degree = A.sum(dim=1).detach().cpu().numpy()

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(h)
    pc1 = pcs[:, 0]

    sp = stats.spearmanr(degree, pc1)

    return {
        "pc1_var": float(pca.explained_variance_ratio_[0]),
        "pc2_var": float(pca.explained_variance_ratio_[1]),
        "spearman_degree_pc1": float(sp.statistic),
        "spearman_degree_pc1_p": float(sp.pvalue),
    }


@torch.no_grad()
def evaluate_pca_diagnostics(model: nn.Module, device: torch.device) -> dict[str, dict[str, float]]:
    """
    Average PCA diagnostics per difficulty regime.
    """
    model.eval()
    out: dict[str, dict[str, float]] = {}

    for regime, (k_min, k_max) in REGIMES.items():
        rows = []

        for _ in range(EVAL_GRAPHS_PER_REGIME):
            k = random.randint(k_min, k_max)
            A, _planted = make_planted_clique(TEST_N, k, device)
            rows.append(pca_degree_diagnostics_one_graph(model, A, k))

        keys = rows[0].keys()
        out[regime] = {key: float(sum(row[key] for row in rows) / len(rows)) for key in keys}

    return out


def print_pca_diagnostics(all_pca: dict[str, dict[str, dict[str, float]]]) -> None:
    print("\n\nPCA / DEGREE DIAGNOSTICS")
    print("========================")
    print("One full-graph GNN pass; PCA on final node embeddings.")
    print("Degree is original row-sum(A). Spearman is between degree and PC1.")
    print()

    for model_name, model_res in all_pca.items():
        print(model_name)
        print("-" * len(model_name))
        print(f"{'regime':<8s} {'PC1 var':>10s} {'PC2 var':>10s} {'Spearman(deg,PC1)':>20s} {'p-value':>10s}")

        for regime in ["easy", "medium", "hard"]:
            m = model_res[regime]
            print(
                f"{regime:<8s} "
                f"{m['pc1_var']:>10.3f} "
                f"{m['pc2_var']:>10.3f} "
                f"{m['spearman_degree_pc1']:>20.3f} "
                f"{m['spearman_degree_pc1_p']:>10.2e}"
            )
        print()


def print_results(all_results: dict[str, dict]) -> None:
    print("\n\nFINAL RESULTS ON n=1000")
    print("=======================")
    print("approx = |found valid clique| / |planted clique|; overlap = planted vertices recovered / planted clique size")
    print()

    for model_name, model_res in all_results.items():
        print(model_name)
        print("-" * len(model_name))
        print(f"{'regime':<8s} {'decoder':<14s} {'approx':>8s} {'overlap':>8s} {'exact':>8s}")

        for regime in ["easy", "medium", "hard"]:
            for decoder in EVAL_DECODERS:
                m = model_res[regime][decoder]
                print(
                    f"{regime:<8s} {decoder:<14s} "
                    f"{m['approx']:>8.3f} {m['overlap']:>8.3f} {m['exact']:>8.3f}"
                )
        print()


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)

    ensure_output_dirs()
    save_run_config()

    device = torch.device(DEVICE)
    print(f"device={device}")
    print(f"TRAIN_N={TRAIN_N}, TEST_N={TEST_N}, p={P_ER}")
    print(f"SHARED_INTERNAL_STEPS={INTERNAL_STEPS}, ACTIVATION={ACTIVATION}, AGGREGATION={AGGREGATION}")
    print("Gradient-aware model injects grad_p clique_objective_loss at every GNN layer.")
    print("No model receives degree as an explicit input feature.")
    print("PC1 decoders use a linear PCA coordinate of final node embeddings at inference.")
    print("Pruned/incremental decoders stop when alive set is a clique, using O(1) incremental clique checks.")
    print("Incremental decoder uses a learned MLP updater, not the fixed degree update.")
    print(f"RUN_PC1_RERUN_DECODER={RUN_PC1_RERUN_DECODER}")
    print(f"EVAL_DECODERS={EVAL_DECODERS}")
    print("Residual-aware sum aggregation is used: removed nodes do not send messages.")
    print("Incremental imitation training uses degree only to define oracle remove-actions, not as a feature or regression target.")
    print("Self-predictor LPR training uses no degree oracle: it mimics the model's own rerun-pruned predictor.")
    print(f"SELF_LPR_STEPS={SELF_LPR_STEPS} means full n-k trajectory when None.")
    print(f"REGIMES={REGIMES}")
    print(f"SKIP_TRAINING={SKIP_TRAINING} (checkpoints from {MODEL_DIR})")
    sanity_check_residual_masking(device)

    specs = [
        TrainSpec("standard_GNN__objective_train", gradient_aware=False, training_way="objective"),
        TrainSpec("gradient_GNN__objective_train", gradient_aware=True, training_way="objective"),
        TrainSpec("standard_GNN__incremental_lpr_imitation_train", gradient_aware=False, training_way="incremental_lpr_imitation"),
        TrainSpec("gradient_GNN__incremental_lpr_imitation_train", gradient_aware=True, training_way="incremental_lpr_imitation"),
        TrainSpec("standard_GNN__self_predictor_lpr_train", gradient_aware=False, training_way="self_predictor_lpr"),
        TrainSpec("gradient_GNN__self_predictor_lpr_train", gradient_aware=True, training_way="self_predictor_lpr"),
    ]

    trained = []
    for spec in specs:
        if SKIP_TRAINING:
            model = load_model_checkpoint(spec, device)
        else:
            model = train_model(spec, device)
            save_model_checkpoint(model, spec)
        trained.append((spec, model))

    all_results = {}
    all_pca = {}
    all_instance_rows: list[dict] = []

    for spec, model in trained:
        if PRINT_EVAL_PROGRESS:
            print(f"\n=== EVALUATING MODEL: {spec.name} ===", flush=True)

        model_results, instance_rows = evaluate_model(model, spec, device)
        all_results[spec.name] = model_results
        all_instance_rows.extend(instance_rows)

        if SAVE_OUTPUTS:
            write_csv(PER_INSTANCE_CSV, all_instance_rows)
            write_csv(AGGREGATE_CSV, aggregate_rows(all_instance_rows))
            print(f"[saved] per-instance results -> {PER_INSTANCE_CSV}", flush=True)
            print(f"[saved] aggregate results -> {AGGREGATE_CSV}", flush=True)

        if PRINT_EVAL_PROGRESS:
            print(f"\n=== PCA DIAGNOSTICS: {spec.name} ===", flush=True)
        all_pca[spec.name] = evaluate_pca_diagnostics(model, device)

    print_results(all_results)
    print_pca_diagnostics(all_pca)

    if SAVE_OUTPUTS:
        write_csv(PER_INSTANCE_CSV, all_instance_rows)
        write_csv(AGGREGATE_CSV, aggregate_rows(all_instance_rows))
        print(f"\nSaved outputs under: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
