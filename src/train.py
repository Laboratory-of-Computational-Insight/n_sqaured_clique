#!/usr/bin/env python3
"""
Train and evaluate k-less SGNN / SGNN+U / SGNN+GU on planted cliques (paper Sec. 3.4).

    python -m src.train --train-regime medium
    python -m src.train --train-regime medium --skip-training

Checkpoints and CSVs under ``runs/sgnn_paper_{easy|medium|hard}/``.
Model: :mod:`models`. SNAP / timing / degree: :mod:`experiments`. See ``README.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
from .gu_plots import save_pca_arch_training_grid_html, save_pca_arch_training_grid_png

from .models import (
    CachedAggregationState,
    CachedFGUState,
    CachedKlessCliqueGradientState,
    GUC_AGG_LAYER,
    ResidualGNN,
    TrainSpec,
    clique_objective_loss_kless,
    masked_removal_ce,
    remove_lowest,
)


# ============================================================
# Config
# ============================================================

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_N = 1000
TEST_N = 1000
P_ER = 0.5

# ===========================================================================
# TRAINING HYPERPARAMETERS (paper: Sec. 3.4)
#
#   PAPER  (default): EPOCHS=64, TRAIN_GRAPHS_PER_EPOCH=1000  → 64,000 total graphs
#   TESTING shortcut: EPOCHS=300, TRAIN_GRAPHS_PER_EPOCH=1    → 300 total graphs
#                     (from sgnn_kless_autograd/run_config.json smoke-test runs)
#
# DO NOT change EPOCHS/TRAIN_GRAPHS_PER_EPOCH for paper reproduction.
# ===========================================================================
EPOCHS = 300
TRAIN_GRAPHS_PER_EPOCH = 1
HIDDEN = 64
INTERNAL_STEPS = 4
LR = 1e-3

OBJECTIVE_WEIGHT = 1.0
SELF_LPR_STEPS = 300  # SIT teacher steps (Sec. 3.3.2)
SELF_LPR_WEIGHT = 1.0

# ===========================================================================
# EVALUATION HYPERPARAMETERS
#
#   PAPER  (default): EVAL_GRAPHS_PER_REGIME=1000
#   TESTING shortcut: EVAL_GRAPHS_PER_REGIME=30
#
# DO NOT change for paper reproduction.
# ===========================================================================
EVAL_GRAPHS_PER_REGIME = 30
PRINT_EVAL_PROGRESS = False
SKIP_TRAINING = False
SAVE_OUTPUTS = True

# Run diagnostics via experiments.py / gu_plots.py (not in main train loop).
RUN_DEGREE_CORR_CHECK = False
DEGREE_CORR_GRAPHS_PER_REGIME = 1000
DEGREE_CORR_DECODER_STEPS = 5
DEGREE_CORR_DECODER = "upr"

RUN_PCA_PLOTS = False
PCA_GRAPH_IDX = 0
PCA_SAVE_PNG = True
ARCH_PLOT_ORDER = ("SGNN", "SGNN+U", "SGNN+GU")
TRAINING_PLOT_ORDER = ("objective", "self_predictor_lpr")

REGIMES = {
    "easy": (62, 100),
    "medium": (36, 61),
    "hard": (20, 35),
}
REGIME_PRINT_ORDER = ("easy", "medium", "hard")
# Main table (Table 2) uses Medium-trained checkpoints (Sec. 3.4).
TRAIN_REGIME = "medium"
TRAIN_K_MIN, TRAIN_K_MAX = REGIMES[TRAIN_REGIME]

RUN_NAME = f"sgnn_paper_{TRAIN_REGIME}"
OUTPUT_DIR = Path("runs") / RUN_NAME
MODEL_DIR = OUTPUT_DIR / "models"
PER_INSTANCE_CSV = OUTPUT_DIR / "per_instance_results.csv"
AGGREGATE_CSV = OUTPUT_DIR / "aggregate_results.csv"
RUN_CONFIG_JSON = OUTPUT_DIR / "run_config.json"
DEGREE_CORR_CSV = OUTPUT_DIR / "upr_priority_degree_corr.csv"
PCA_PLOT_DIR = OUTPUT_DIR / "plots" / "pca"
SNAP_GT_CACHE_DIR = OUTPUT_DIR / "snap_gt_cache"
SNAP_PER_INSTANCE_CSV = OUTPUT_DIR / "snap_per_instance_results.csv"
SNAP_AGGREGATE_CSV = OUTPUT_DIR / "snap_aggregate_results.csv"

# name, updater_type, use_gradient_in_updater, neighbor_gate_updater
ARCHITECTURES: list[tuple[str, str, bool, bool]] = [
    ("SGNN", "none", False, False),
    ("SGNN+U", "U", False, True),
    ("SGNN+GU", "GU", True, True),
    ("SGNN+DGU", "DGU", True, True),
    ("SGNN+GUL", "GUL", True, True),
    ("SGNN+GU-BA", "GU", True, True),
    ("SGNN+GUC", "GUC", True, True),
    ("SGNN+FGU", "FGU", True, True),
]

TRAINING_BY_ARCH: dict[str, tuple[str, ...]] = {
    "SGNN": ("objective",),
    "SGNN+U": ("objective", "self_predictor_lpr"),
    "SGNN+GU": ("objective", "self_predictor_lpr"),
    "SGNN+DGU": ("objective", "self_predictor_lpr"),
    "SGNN+GUL": ("objective", "self_predictor_lpr"),
    # Option A: ER objective (keeps GNN encoder stable) + mixed ER/BA SIT (exposes GU MLP to both regimes)
    "SGNN+GU-BA": ("objective", "ba_self_predictor_lpr"),
    "SGNN+GUC": ("objective", "self_predictor_lpr"),
    "SGNN+FGU": ("objective", "self_predictor_lpr"),
}

DECODERS_BY_ARCH: dict[str, tuple[str, ...]] = {
    "SGNN": ("one_pass", "rerun_pruned", "lpr", "pgu", "lpgu"),
    "SGNN+U": ("one_pass", "rerun_pruned", "lpr", "upr", "pgu", "lpgu"),
    "SGNN+GU": ("one_pass", "rerun_pruned", "lpr", "upr", "pgu", "lpgu"),
    "SGNN+DGU": ("upr",),
    "SGNN+GUL": ("upr",),
    "SGNN+GU-BA": ("upr",),
    "SGNN+GUC": ("upr",),
    "SGNN+FGU": ("upr",),
}

ALL_DECODERS = ("one_pass", "rerun_pruned", "lpr", "upr")


# ============================================================
# Output utilities
# ============================================================

def apply_run_config(*, train_regime: str, layers: int | None = None, epochs: int | None = None) -> None:
    """Point outputs and training k-range at runs/sgnn_paper_{easy|medium|hard}/."""
    global TRAIN_REGIME, TRAIN_K_MIN, TRAIN_K_MAX, RUN_NAME, INTERNAL_STEPS, EPOCHS
    global OUTPUT_DIR, MODEL_DIR, PER_INSTANCE_CSV, AGGREGATE_CSV, RUN_CONFIG_JSON
    global DEGREE_CORR_CSV, PCA_PLOT_DIR, SNAP_GT_CACHE_DIR
    global SNAP_PER_INSTANCE_CSV, SNAP_AGGREGATE_CSV

    if train_regime not in REGIMES:
        raise ValueError(f"train_regime must be one of {tuple(REGIMES)}, got {train_regime!r}")

    TRAIN_REGIME = train_regime
    TRAIN_K_MIN, TRAIN_K_MAX = REGIMES[train_regime]
    RUN_NAME = f"sgnn_paper_{train_regime}"
    if layers is not None:
        INTERNAL_STEPS = layers
        RUN_NAME = f"{RUN_NAME}_L{layers}"
    if epochs is not None:
        EPOCHS = epochs
        RUN_NAME = f"{RUN_NAME}_E{epochs}"
    OUTPUT_DIR = Path("runs") / RUN_NAME
    MODEL_DIR = OUTPUT_DIR / "models"
    PER_INSTANCE_CSV = OUTPUT_DIR / "per_instance_results.csv"
    AGGREGATE_CSV = OUTPUT_DIR / "aggregate_results.csv"
    RUN_CONFIG_JSON = OUTPUT_DIR / "run_config.json"
    DEGREE_CORR_CSV = OUTPUT_DIR / "upr_priority_degree_corr.csv"
    PCA_PLOT_DIR = OUTPUT_DIR / "plots" / "pca"
    SNAP_GT_CACHE_DIR = OUTPUT_DIR / "snap_gt_cache"
    SNAP_PER_INSTANCE_CSV = OUTPUT_DIR / "snap_per_instance_results.csv"
    SNAP_AGGREGATE_CSV = OUTPUT_DIR / "snap_aggregate_results.csv"


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
        "TRAIN_GRAPHS_PER_EPOCH": TRAIN_GRAPHS_PER_EPOCH,
        "TRAIN_REGIME": TRAIN_REGIME,
        "HIDDEN": HIDDEN,
        "INTERNAL_STEPS": INTERNAL_STEPS,
        "LR": LR,
        "OBJECTIVE_WEIGHT": OBJECTIVE_WEIGHT,
        "SELF_LPR_STEPS": SELF_LPR_STEPS,
        "SELF_LPR_WEIGHT": SELF_LPR_WEIGHT,
        "EVAL_GRAPHS_PER_REGIME": EVAL_GRAPHS_PER_REGIME,
        "REGIMES": REGIMES,
        "ARCHITECTURES": ARCHITECTURES,
        "TRAINING_BY_ARCH": TRAINING_BY_ARCH,
        "DECODERS_BY_ARCH": DECODERS_BY_ARCH,
        "K_USAGE_NOTE": "k is used only for graph generation and evaluation reporting, not model/loss/GU/decoder.",
        "RUN_DEGREE_CORR_CHECK": RUN_DEGREE_CORR_CHECK,
        "DEGREE_CORR_GRAPHS_PER_REGIME": DEGREE_CORR_GRAPHS_PER_REGIME,
        "DEGREE_CORR_DECODER_STEPS": DEGREE_CORR_DECODER_STEPS,
        "RUN_PCA_PLOTS": RUN_PCA_PLOTS,
        "PCA_PLOT_DIR": str(PCA_PLOT_DIR),
        "PCA_GRAPH_IDX": PCA_GRAPH_IDX,
        "PCA_GRID": "rows=SGNN,SGNN+U,SGNN+GU; cols=objective,PR; one graph per eval regime",
        "ARCH_PLOT_ORDER": ARCH_PLOT_ORDER,
        "TRAINING_PLOT_ORDER": TRAINING_PLOT_ORDER,
    }

    with open(RUN_CONFIG_JSON, "w") as f:
        json.dump(config, f, indent=2)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not SAVE_OUTPUTS or not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean_std_list(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), std


def _fmt_pm(mean: float, std: float, decimals: int = 3) -> str:
    if mean != mean:
        return "n/a"
    if std != std or std < 1e-12:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f}±{std:.{decimals}f}"


def aggregate_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row["model"],
            row["architecture"],
            row["updater_type"],
            row["use_gradient_in_updater"],
            row["neighbor_gate_updater"],
            row["training_way"],
            row["regime"],
            row["decoder"],
        )
        groups.setdefault(key, []).append(row)

    out: list[dict] = []
    for key, group in groups.items():
        (
            model,
            architecture,
            updater_type,
            use_gradient_in_updater,
            neighbor_gate_updater,
            training_way,
            regime,
            decoder,
        ) = key
        approx_m, approx_s = _mean_std_list([r["approx"] for r in group])
        overlap_m, overlap_s = _mean_std_list([r["overlap"] for r in group])
        planted_exact_m, planted_exact_s = _mean_std_list([r["planted_exact"] for r in group])
        clique_at_least_k_m, clique_at_least_k_s = _mean_std_list([r["clique_at_least_k"] for r in group])
        size_m, size_s = _mean_std_list([r["found_size"] for r in group])
        seconds_m, seconds_s = _mean_std_list([r["seconds"] for r in group])
        out.append({
            "model": model,
            "architecture": architecture,
            "updater_type": updater_type,
            "use_gradient_in_updater": use_gradient_in_updater,
            "neighbor_gate_updater": neighbor_gate_updater,
            "training_way": training_way,
            "regime": regime,
            "decoder": decoder,
            "num_instances": len(group),
            "approx_mean": approx_m,
            "approx_std": approx_s,
            "overlap_mean": overlap_m,
            "overlap_std": overlap_s,
            "planted_exact_mean": planted_exact_m,
            "planted_exact_std": planted_exact_s,
            "clique_at_least_k_mean": clique_at_least_k_m,
            "clique_at_least_k_std": clique_at_least_k_s,
            "found_size_mean": size_m,
            "found_size_std": size_s,
            "seconds_mean": seconds_m,
            "seconds_std": seconds_s,
        })
    return out


# ============================================================
# Data
# ============================================================

def make_planted_clique(n: int, k: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate data. k is allowed here because it defines the synthetic distribution."""
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
    """k is returned only for logging/debugging. Do not pass it to loss/model/decoder."""
    k = random.randint(TRAIN_K_MIN, TRAIN_K_MAX)
    A, planted = make_planted_clique(TRAIN_N, k, device)
    return A, planted, k


_BA_M = 5  # preferential-attachment edges per new node → avg degree ≈ 10


def _generate_ba_adj_numpy(n: int, m: int) -> np.ndarray:
    """Pure-numpy Barabási-Albert preferential-attachment graph (symmetric, no self-loops)."""
    adj = np.zeros((n, n), dtype=np.float32)
    deg = np.zeros(n, dtype=np.float64)
    init = min(m + 1, n)
    for i in range(init):
        for j in range(i + 1, init):
            adj[i, j] = adj[j, i] = 1.0
            deg[i] += 1.0
            deg[j] += 1.0
    for new_v in range(init, n):
        deg_sum = deg[:new_v].sum()
        probs = deg[:new_v] / deg_sum if deg_sum > 0 else np.ones(new_v) / new_v
        targets = np.random.choice(new_v, size=min(m, new_v), replace=False, p=probs)
        for t in targets:
            adj[new_v, t] = adj[t, new_v] = 1.0
            deg[new_v] += 1.0
            deg[t] += 1.0
    return adj


def sample_ba_train_graph(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
    """BA(n, m=5) graph with planted clique. Sparse, power-law degree distribution."""
    k = random.randint(TRAIN_K_MIN, TRAIN_K_MAX)
    n = TRAIN_N
    adj = _generate_ba_adj_numpy(n, _BA_M)
    clique_idx = np.random.choice(n, size=k, replace=False)
    planted = np.zeros(n, dtype=bool)
    planted[clique_idx] = True
    ci = clique_idx
    adj[ci[:, None], ci[None, :]] = 1.0
    np.fill_diagonal(adj, 0.0)
    A = torch.tensor(adj, dtype=torch.float32, device=device)
    planted_t = torch.tensor(planted, dtype=torch.bool, device=device)
    return A, planted_t, k


def sample_ba_mixed_train_graph(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int, str]:
    """50 % ER, 50 % BA. Returns (A, planted, k, graph_type_str) for logging."""
    if random.random() < 0.5:
        A, planted, k = sample_train_graph(device)
        return A, planted, k, "ER"
    A, planted, k = sample_ba_train_graph(device)
    return A, planted, k, "BA"


# ============================================================
# Clique metrics and pruning state
# ============================================================

def is_clique(A: torch.Tensor, selected: torch.Tensor) -> bool:
    idx = selected.nonzero(as_tuple=False).flatten()
    size = int(idx.numel())
    if size <= 1:
        return True
    sub = A[idx[:, None], idx[None, :]]
    return float(sub.sum().item()) == size * (size - 1)


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


def init_alive_degrees_and_edges(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    deg_alive = A.sum(dim=1)
    alive_edges = A.sum() / 2.0
    return deg_alive, alive_edges


def alive_is_clique_fast(alive_count: int, alive_edges: torch.Tensor) -> bool:
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
    v_int = int(v.item())
    alive_next = alive.clone()
    alive_next[v_int] = False

    alive_edges_next = alive_edges - deg_alive[v_int]
    deg_alive_next = deg_alive - A[:, v_int]
    deg_alive_next[v_int] = 0.0
    deg_alive_next = deg_alive_next * alive_next.float()

    return alive_next, deg_alive_next, alive_edges_next


def can_add_to_clique(A: torch.Tensor, clique: torch.Tensor, v: torch.Tensor) -> bool:
    idx = clique.nonzero(as_tuple=False).flatten()
    if idx.numel() == 0:
        return True
    return bool((A[v, idx] > 0.5).all().item())


def add_back_removed_nodes(A: torch.Tensor, alive_clique: torch.Tensor, removed: list[torch.Tensor]) -> torch.Tensor:
    clique = alive_clique.clone()
    for v in reversed(removed):
        if can_add_to_clique(A, clique, v):
            clique[v] = True
    return clique


def approx_ratio(found_clique: torch.Tensor, planted: torch.Tensor) -> float:
    true_k = int(planted.sum().item())
    return 0.0 if true_k == 0 else float(found_clique.sum().item()) / true_k


def planted_overlap(found_clique: torch.Tensor, planted: torch.Tensor) -> float:
    true_k = int(planted.sum().item())
    return 0.0 if true_k == 0 else float((found_clique & planted).sum().item()) / true_k


def planted_exact(found_clique: torch.Tensor, planted: torch.Tensor) -> float:
    return float(int((found_clique & planted).sum().item()) == int(planted.sum().item()))


def clique_at_least_planted_size(found_clique: torch.Tensor, planted: torch.Tensor, A: torch.Tensor) -> float:
    true_k = int(planted.sum().item())
    return float(int(found_clique.sum().item()) >= true_k and is_clique(A, found_clique))


# ============================================================
# K-less clique objective and cached GU gradient
# ============================================================


# ============================================================
# Training losses
# ============================================================

def _get_grad(cache: "CachedKlessCliqueGradientState | None", updater_type: str) -> "torch.Tensor | None":
    """Return gradient from cache, applying DGU renormalization if requested."""
    if cache is None:
        return None
    return cache.gradient_dgu() if updater_type == "DGU" else cache.gradient()

def self_predictor_lpr_loss(model: ResidualGNN, A: torch.Tensor, steps: int | None) -> torch.Tensor:
    """
    K-less PR loss.

    Student UPR imitates the model's rerun-pruned deletion choices.
    Stop condition: residual graph is a clique (no k needed).
    """
    if model.updater_type == "none":
        return torch.tensor(0.0, device=A.device)

    n = A.shape[0]
    alive_student = torch.ones(n, dtype=torch.bool, device=A.device)
    alive_teacher = torch.ones(n, dtype=torch.bool, device=A.device)
    deg_alive, alive_edges = init_alive_degrees_and_edges(A)
    alive_count = n

    if model.updater_type == "GUC":
        with torch.no_grad():
            _, H_frozen = model.embed_with_intermediate(A, alive_student, GUC_AGG_LAYER)
        student_scores = model.normal_scores(A, alive_student)
    else:
        student_scores = model.normal_scores(A, alive_student)
        H_frozen = None

    grad_cache = None
    if model.use_gradient_in_updater:
        grad_cache = CachedKlessCliqueGradientState.build(A, student_scores, alive_student)

    agg_cache = None
    if model.updater_type == "GUC" and H_frozen is not None:
        agg_cache = CachedAggregationState.build(A, H_frozen, alive_student)

    fgu_cache = None
    if model.updater_type == "FGU":
        fgu_cache = CachedFGUState.build(model, A, alive_student)

    losses: list[torch.Tensor] = []
    max_steps = (n - 1) if steps is None else min(steps, n - 1)

    for _ in range(max_steps):
        if alive_count <= 1 or alive_is_clique_fast(alive_count, alive_edges):
            break

        with torch.no_grad():
            teacher_scores = model.normal_scores(A, alive_teacher)
            target = remove_lowest(teacher_scores, alive_teacher)

        losses.append(masked_removal_ce(student_scores, alive_student, target))

        alive_teacher = alive_teacher.clone()
        alive_teacher[target] = False

        grad = _get_grad(grad_cache, model.updater_type)

        agg_channel = agg_cache.channel() if agg_cache is not None else None
        fgu_vec = fgu_cache.fgu_signal(target, A) if fgu_cache is not None else None

        alive_student, deg_alive, alive_edges = remove_vertex_update_state(
            A, alive_student, deg_alive, alive_edges, target
        )
        alive_count -= 1

        if grad_cache is not None:
            grad_cache.delete_vertex_only(target)
        if agg_cache is not None:
            agg_cache.delete_vertex_only(target)
        if fgu_cache is not None:
            fgu_cache.delete_vertex_only(target)

        student_scores = model.update_scores(
            scores=student_scores,
            A=A,
            alive_after=alive_student,
            removed=target,
            grad=grad,
            agg_channel=agg_channel,
            fgu_vec=fgu_vec,
        )

    if not losses:
        return torch.tensor(0.0, device=A.device)
    return torch.stack(losses).mean()


def train_loss(model: ResidualGNN, A: torch.Tensor, training_way: str) -> torch.Tensor:
    alive = torch.ones(A.shape[0], dtype=torch.bool, device=A.device)
    obj = clique_objective_loss_kless(model.normal_scores(A, alive), A, alive)

    # Strip "ba_" prefix — graph sampling is handled by the caller; loss logic is identical
    base_way = training_way[3:] if training_way.startswith("ba_") else training_way

    if base_way == "objective":
        return obj

    if base_way == "self_predictor_lpr":
        pr = self_predictor_lpr_loss(model, A, SELF_LPR_STEPS)
        # GUL: upweight SIT so the linear updater learns from decoder feedback, not just objective
        sit_w = 5.0 if model.updater_type == "GUL" else SELF_LPR_WEIGHT
        return OBJECTIVE_WEIGHT * obj + sit_w * pr

    raise ValueError(training_way)


# ============================================================
# Decoders: no k input
# ============================================================

@torch.no_grad()
def select_one_pass(model: ResidualGNN, A: torch.Tensor) -> tuple[torch.Tensor, int]:
    n = A.shape[0]
    alive = torch.ones(n, dtype=torch.bool, device=A.device)
    scores = model.normal_scores(A, alive)
    order = scores.argsort(descending=True)
    found = greedy_clique_from_order(A, order, max_k=None)
    return found, n - int(found.sum().item())


@torch.no_grad()
def select_rerun_pruned(model: ResidualGNN, A: torch.Tensor) -> tuple[torch.Tensor, int]:
    n = A.shape[0]
    alive = torch.ones(n, dtype=torch.bool, device=A.device)
    deg_alive, alive_edges = init_alive_degrees_and_edges(A)
    alive_count = n
    removed: list[torch.Tensor] = []

    while not alive_is_clique_fast(alive_count, alive_edges):
        scores = model.normal_scores(A, alive)
        v = remove_lowest(scores, alive)
        removed.append(v)
        alive, deg_alive, alive_edges = remove_vertex_update_state(A, alive, deg_alive, alive_edges, v)
        alive_count -= 1

    return add_back_removed_nodes(A, alive, removed), len(removed)


@torch.no_grad()
def select_lpr(model: ResidualGNN, A: torch.Tensor) -> tuple[torch.Tensor, int]:
    n = A.shape[0]
    alive = torch.ones(n, dtype=torch.bool, device=A.device)
    deg_alive, alive_edges = init_alive_degrees_and_edges(A)
    alive_count = n
    scores = model.normal_scores(A, alive)
    removed: list[torch.Tensor] = []

    while not alive_is_clique_fast(alive_count, alive_edges):
        v = remove_lowest(scores, alive)
        removed.append(v)
        alive, deg_alive, alive_edges = remove_vertex_update_state(A, alive, deg_alive, alive_edges, v)
        alive_count -= 1
        scores = scores.masked_fill(~alive.bool(), -1e9)

    return add_back_removed_nodes(A, alive, removed), len(removed)


@torch.no_grad()
def select_upr(model: ResidualGNN, A: torch.Tensor) -> torch.Tensor:
    if model.updater_type == "none":
        raise RuntimeError("upr requires SGNN+U or SGNN+GU")

    n = A.shape[0]
    alive = torch.ones(n, dtype=torch.bool, device=A.device)
    deg_alive, alive_edges = init_alive_degrees_and_edges(A)
    alive_count = n
    if model.updater_type == "GUC":
        _, H_frozen = model.embed_with_intermediate(A, alive, GUC_AGG_LAYER)
        scores = model.normal_scores(A, alive)
    else:
        scores = model.normal_scores(A, alive)
        H_frozen = None

    grad_cache = None
    if model.use_gradient_in_updater:
        grad_cache = CachedKlessCliqueGradientState.build(A, scores, alive)

    agg_cache = None
    if model.updater_type == "GUC" and H_frozen is not None:
        agg_cache = CachedAggregationState.build(A, H_frozen, alive)

    fgu_cache = None
    if model.updater_type == "FGU":
        fgu_cache = CachedFGUState.build(model, A, alive)

    removed: list[torch.Tensor] = []
    while not alive_is_clique_fast(alive_count, alive_edges):
        grad = _get_grad(grad_cache, model.updater_type)
        agg_channel = agg_cache.channel() if agg_cache is not None else None
        v = remove_lowest(scores, alive)
        removed.append(v)

        fgu_vec = fgu_cache.fgu_signal(v, A) if fgu_cache is not None else None

        alive, deg_alive, alive_edges = remove_vertex_update_state(A, alive, deg_alive, alive_edges, v)
        alive_count -= 1

        if grad_cache is not None:
            grad_cache.delete_vertex_only(v)
        if agg_cache is not None:
            agg_cache.delete_vertex_only(v)
        if fgu_cache is not None:
            fgu_cache.delete_vertex_only(v)

        scores = model.update_scores(
            scores=scores,
            A=A,
            alive_after=alive,
            removed=v,
            grad=grad,
            agg_channel=agg_channel,
            fgu_vec=fgu_vec,
        )

    return add_back_removed_nodes(A, alive, removed), len(removed)


@torch.no_grad()
def select_pgu(model: ResidualGNN, A: torch.Tensor, refresh_every: int | None = None) -> tuple[torch.Tensor, int]:
    """Periodic refresh: full GNN re-run every sqrt(n) steps, updater (if any) between refreshes.

    Works for all architectures: SGNN (stale scores between refreshes), SGNN+U, SGNN+GU.
    Each re-run sees a smaller residual graph — the detection problem gets progressively
    easier (k/n_alive grows), so late re-runs find the clique even on hub graphs.
    Cost: O(sqrt(n)) re-runs × O(n²) = O(n^2.5) total.
    """
    n = A.shape[0]
    if refresh_every is None:
        refresh_every = max(1, int(n ** 0.5))

    alive = torch.ones(n, dtype=torch.bool, device=A.device)
    deg_alive, alive_edges = init_alive_degrees_and_edges(A)
    alive_count = n
    scores = model.normal_scores(A, alive)

    grad_cache = None
    if model.use_gradient_in_updater:
        grad_cache = CachedKlessCliqueGradientState.build(A, scores, alive)

    removed: list[torch.Tensor] = []
    steps_since_refresh = 0

    while not alive_is_clique_fast(alive_count, alive_edges):
        if steps_since_refresh >= refresh_every:
            scores = model.normal_scores(A, alive)
            if grad_cache is not None:
                grad_cache = CachedKlessCliqueGradientState.build(A, scores, alive)
            steps_since_refresh = 0

        grad = _get_grad(grad_cache, model.updater_type)
        v = remove_lowest(scores, alive)
        removed.append(v)

        alive, deg_alive, alive_edges = remove_vertex_update_state(A, alive, deg_alive, alive_edges, v)
        alive_count -= 1

        if grad_cache is not None:
            grad_cache.delete_vertex_only(v)

        scores = model.update_scores(
            scores=scores, A=A, alive_after=alive, removed=v, grad=grad
        )
        steps_since_refresh += 1

    return add_back_removed_nodes(A, alive, removed), len(removed)


@torch.no_grad()
def select_lpgu(model: ResidualGNN, A: torch.Tensor) -> tuple[torch.Tensor, int]:
    """LPGU: log(n) recalculations total, refresh every n/log2(n) steps.

    Cheaper than PGU (O(n² log n) vs O(n^2.5)) while still getting periodic
    structural resets. Useful when n is large.
    """
    import math
    n = A.shape[0]
    refresh_every = max(1, int(n / math.log2(max(n, 2))))
    return select_pgu(model, A, refresh_every=refresh_every)


@torch.no_grad()
def select_ldr_decoder(A: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Least-Degree Removal: remove minimum-residual-degree vertex until a clique remains.

    Runs entirely in numpy on CPU — faster than GPU for the small graphs typical in SNAP.
    """
    adj = A.cpu().numpy()
    n = adj.shape[0]
    alive = np.ones(n, dtype=bool)
    deg = adj.sum(axis=1)          # residual degrees (float)
    alive_edges = deg.sum() / 2.0
    alive_count = n
    removed: list[int] = []

    while alive_count > 1:
        target = alive_count * (alive_count - 1) / 2.0
        if alive_edges == target:
            break
        deg_masked = np.where(alive, deg, np.inf)
        v = int(np.argmin(deg_masked))
        removed.append(v)
        alive_edges -= deg[v]
        deg -= adj[:, v]
        deg[v] = 0.0
        deg *= alive.astype(np.float32)
        alive[v] = False
        alive_count -= 1

    # add back removed vertices that can rejoin the clique
    clique = alive.copy()
    for v in reversed(removed):
        idx = np.where(clique)[0]
        if idx.size == 0 or np.all(adj[v, idx] > 0.5):
            clique[v] = True

    found = torch.tensor(clique, dtype=torch.bool, device=A.device)
    return found, len(removed)


@torch.no_grad()
def eval_one_decoder(
    model: ResidualGNN | None,
    A: torch.Tensor,
    planted: torch.Tensor,
    decoder: str,
) -> dict[str, float]:
    if decoder == "ldr":
        found, steps = select_ldr_decoder(A)
    elif decoder == "pgu":
        found, steps = select_pgu(model, A)
    elif decoder == "lpgu":
        found, steps = select_lpgu(model, A)
    elif decoder == "one_pass":
        found, steps = select_one_pass(model, A)
    elif decoder == "rerun_pruned":
        found, steps = select_rerun_pruned(model, A)
    elif decoder == "lpr":
        found, steps = select_lpr(model, A)
    elif decoder == "upr":
        found, steps = select_upr(model, A)
    else:
        raise ValueError(decoder)

    return {
        "approx": approx_ratio(found, planted),
        "overlap": planted_overlap(found, planted),
        "planted_exact": planted_exact(found, planted),
        "clique_at_least_k": clique_at_least_planted_size(found, planted, A),
        "found_size": float(found.sum().item()),
        "is_clique": float(is_clique(A, found)),
        "stopping_steps": float(steps),
    }


# ============================================================
# UPR priority vs residual degree (short correlation check)
# ============================================================

def residual_degree_numpy(A: torch.Tensor, alive: torch.Tensor) -> np.ndarray:
    return (A * alive.float()[None, :]).sum(dim=1).detach().cpu().numpy()


def _alive_sub(values: np.ndarray, alive: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)[alive.astype(bool)]


def _spearman_with_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan"), float("nan")
    result = stats.spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def _pearson_with_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan"), float("nan")
    result = stats.pearsonr(x, y)
    return float(result.statistic), float(result.pvalue)


def degree_corr_seed(spec: "TrainSpec", regime: str, graph_idx: int) -> int:
    return SEED + (hash(spec.name) % 10_000) + (hash(regime) % 1_000) + graph_idx * 97


@torch.no_grad()
def collect_upr_priority_degree_corr(
    model: ResidualGNN,
    A: torch.Tensor,
    *,
    max_steps: int,
) -> list[dict]:
    n = A.shape[0]
    alive = torch.ones(n, dtype=torch.bool, device=A.device)
    deg_alive, alive_edges = init_alive_degrees_and_edges(A)
    alive_count = n

    if model.updater_type == "GUC":
        _, H_frozen = model.embed_with_intermediate(A, alive, GUC_AGG_LAYER)
        scores = model.normal_scores(A, alive)
    else:
        scores = model.normal_scores(A, alive)
        H_frozen = None

    grad_cache = None
    if model.use_gradient_in_updater:
        grad_cache = CachedKlessCliqueGradientState.build(A, scores, alive)

    agg_cache = None
    if model.updater_type == "GUC" and H_frozen is not None:
        agg_cache = CachedAggregationState.build(A, H_frozen, alive)

    fgu_cache = None
    if model.updater_type == "FGU":
        fgu_cache = CachedFGUState.build(model, A, alive)

    rows: list[dict] = []

    for step in range(max_steps):
        if alive_count <= 1 or alive_is_clique_fast(alive_count, alive_edges):
            break

        alive_np = alive.detach().cpu().numpy().astype(bool)
        scores_before = scores.clone()
        degree_full = residual_degree_numpy(A, alive)
        priority_full = scores_before.detach().cpu().numpy()
        degree_sub = _alive_sub(degree_full, alive_np)
        priority_sub = _alive_sub(priority_full, alive_np)

        sp, sp_p = _spearman_with_p(priority_sub, degree_sub)
        pr, pr_p = _pearson_with_p(priority_sub, degree_sub)
        rows.append({
            "step": step,
            "state": "removal_priority",
            "alive_count": int(alive_count),
            "spearman": sp,
            "abs_spearman": abs(sp) if sp == sp else float("nan"),
            "spearman_p": sp_p,
            "pearson": pr,
            "pearson_p": pr_p,
        })

        grad = _get_grad(grad_cache, model.updater_type)
        agg_channel = agg_cache.channel() if agg_cache is not None else None
        v = remove_lowest(scores, alive)
        fgu_vec = fgu_cache.fgu_signal(v, A) if fgu_cache is not None else None
        alive, deg_alive, alive_edges = remove_vertex_update_state(A, alive, deg_alive, alive_edges, v)
        alive_count -= 1
        if grad_cache is not None:
            grad_cache.delete_vertex_only(v)
        if agg_cache is not None:
            agg_cache.delete_vertex_only(v)
        if fgu_cache is not None:
            fgu_cache.delete_vertex_only(v)

        scores = model.update_scores(
            scores=scores_before,
            A=A,
            alive_after=alive,
            removed=v,
            grad=grad,
            agg_channel=agg_channel,
            fgu_vec=fgu_vec,
        )

        alive_np_after = alive.detach().cpu().numpy().astype(bool)
        degree_after = _alive_sub(residual_degree_numpy(A, alive), alive_np_after)
        updated_sub = _alive_sub(scores.detach().cpu().numpy(), alive_np_after)
        sp2, sp_p2 = _spearman_with_p(updated_sub, degree_after)
        pr2, pr_p2 = _pearson_with_p(updated_sub, degree_after)
        rows.append({
            "step": step,
            "state": "updated_priority",
            "alive_count": int(alive_count),
            "spearman": sp2,
            "abs_spearman": abs(sp2) if sp2 == sp2 else float("nan"),
            "spearman_p": sp_p2,
            "pearson": pr2,
            "pearson_p": pr_p2,
        })

    return rows


@torch.no_grad()
def run_degree_corr_diagnostics(
    model: ResidualGNN,
    spec: "TrainSpec",
    device: torch.device,
) -> list[dict]:
    if DEGREE_CORR_DECODER != "upr" or "upr" not in DECODERS_BY_ARCH.get(spec.architecture, ()):
        return []

    all_rows: list[dict] = []
    for regime in REGIME_PRINT_ORDER:
        k_min, k_max = REGIMES[regime]
        for graph_idx in range(DEGREE_CORR_GRAPHS_PER_REGIME):
            seed = degree_corr_seed(spec, regime, graph_idx)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            k_gen = random.randint(k_min, k_max)
            A, _planted = make_planted_clique(TEST_N, k_gen, device)
            trace = collect_upr_priority_degree_corr(model, A, max_steps=DEGREE_CORR_DECODER_STEPS)
            for row in trace:
                all_rows.append({
                    "model": spec.name,
                    "regime": regime,
                    "graph_idx": graph_idx,
                    "instance_seed": seed,
                    "n": TEST_N,
                    "k_for_generation_only": k_gen,
                    "decoder": DEGREE_CORR_DECODER,
                    **row,
                })
    return all_rows


def print_degree_corr_summary(rows: list[dict]) -> None:
    if not rows:
        return

    print("\n" + "=" * 88)
    print(
        f"UPR priority vs residual degree (alive only; "
        f"n={DEGREE_CORR_GRAPHS_PER_REGIME} graphs/regime; steps 0–{DEGREE_CORR_DECODER_STEPS - 1})"
    )
    print("removal_priority = scores for remove_lowest; updated_priority = after update_scores")
    print("=" * 88)

    models = sorted({r["model"] for r in rows})
    for model in models:
        print(model)
        for state in ("removal_priority", "updated_priority"):
            print(f"  [{state}]  mean signed ρ / mean |ρ| by step (E M H)")
            for step in range(DEGREE_CORR_DECODER_STEPS):
                parts: list[str] = []
                for regime in REGIME_PRINT_ORDER:
                    subset = [
                        r for r in rows
                        if r["model"] == model
                        and r["state"] == state
                        and r["regime"] == regime
                        and r["step"] == step
                    ]
                    if not subset:
                        parts.append(f"{regime}:n/a")
                        continue
                    sp = np.asarray([float(r["spearman"]) for r in subset], dtype=np.float64)
                    ab = np.asarray([float(r["abs_spearman"]) for r in subset], dtype=np.float64)
                    parts.append(f"{regime}:{np.nanmean(sp):+.2f}/{np.nanmean(ab):.2f}")
                print(f"    step {step:>2d}:  " + "  ".join(parts))
        print()


# ============================================================
# Full-graph PCA plots (SGNN × SGNN+U × SGNN+GU × training)
# ============================================================

def pca_plot_seed(regime: str, graph_idx: int) -> int:
    return SEED + (hash(regime) % 10_000) + graph_idx * 97


def _training_plot_label(training_way: str) -> str:
    return "objective" if training_way == "objective" else "PR"


@torch.no_grad()
def gnn_hidden_pca_full_graph(model: ResidualGNN, A: torch.Tensor) -> dict[str, np.ndarray | float]:
    """One forward on full A (all nodes); PCA on final hidden; degree = row-sum(A)."""
    n = A.shape[0]
    alive = torch.ones(n, dtype=torch.bool, device=A.device)
    h, _logits, _probs = model.hidden_logits_probs(A, alive)
    degree = A.sum(dim=1).detach().cpu().numpy().astype(np.float64)
    h_np = h.detach().cpu().numpy()

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(h_np)
    pc1 = pcs[:, 0].copy()
    pc2 = pcs[:, 1].copy()

    sp, _sp_p = _spearman_with_p(pc1, degree)
    if sp == sp and sp < 0:
        pc1 = -pc1

    sp_final, sp_p = _spearman_with_p(pc1, degree)
    pr_final, pr_p = _pearson_with_p(pc1, degree)

    return {
        "pc1": pc1,
        "pc2": pc2,
        "degree": degree,
        "pc1_var": float(pca.explained_variance_ratio_[0]),
        "pc2_var": float(pca.explained_variance_ratio_[1]),
        "spearman_pc1_degree": sp_final,
        "spearman_p": sp_p,
        "pearson_pc1_degree": pr_final,
        "pearson_p": pr_p,
    }


def _pca_scatter_trace(data: dict, *, show_colorbar: bool) -> "object":
    import plotly.graph_objects as go

    marker: dict = {
        "size": 6,
        "color": data["degree"],
        "colorscale": "Turbo",
        "opacity": 0.85,
        "line": {"color": "rgba(0,0,0,0.2)", "width": 0.35},
    }
    if show_colorbar:
        marker["colorbar"] = {
            "title": {"text": "Degree", "font": {"size": 11}},
            "tickfont": {"size": 10},
            "thickness": 14,
            "len": 0.72,
        }
    return go.Scatter(
        x=data["pc1"],
        y=data["pc2"],
        mode="markers",
        marker=marker,
        hovertemplate="PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>degree: %{marker.color}<extra></extra>",
    )


def plot_pca_single_html(
    data: dict,
    *,
    title: str,
    out_path: Path,
) -> None:
    import plotly.graph_objects as go

    pc1_var = 100.0 * float(data["pc1_var"])
    pc2_var = 100.0 * float(data["pc2_var"])
    rho = float(data["spearman_pc1_degree"])
    fig = go.Figure(_pca_scatter_trace(data, show_colorbar=True))
    fig.update_layout(
        template="plotly_white",
        title={
            "text": (
                f"<b>{title}</b><br>"
                f"<sup>PC1={pc1_var:.1f}% PC2={pc2_var:.1f}% · "
                f"Spearman(deg,PC1)={rho:+.3f}</sup>"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        width=720,
        height=560,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#222"),
        xaxis_title=f"PC1 ({pc1_var:.1f}% var.)",
        yaxis_title=f"PC2 ({pc2_var:.1f}% var.)",
        margin=dict(l=56, r=40, t=72, b=52),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def _scatter_pca_panel(ax, data: dict):
    from matplotlib.colors import Normalize

    pc1 = data["pc1"]
    pc2 = data["pc2"]
    degree = data["degree"]
    norm = Normalize(vmin=float(degree.min()), vmax=float(degree.max()))
    sc = ax.scatter(
        pc1,
        pc2,
        c=degree,
        cmap="turbo",
        norm=norm,
        s=14,
        alpha=0.85,
        linewidths=0.2,
        edgecolors="0.35",
    )
    pc1_var = 100.0 * float(data["pc1_var"])
    pc2_var = 100.0 * float(data["pc2_var"])
    rho = float(data["spearman_pc1_degree"])
    ax.set_xlabel(f"PC1 ({pc1_var:.0f}%)", fontsize=9)
    ax.set_ylabel(f"PC2 ({pc2_var:.0f}%) · ρ={rho:+.2f}", fontsize=9)
    ax.set_facecolor("white")
    ax.grid(True, color=(0, 0, 0, 0.1), linewidth=0.5)
    return sc


def plot_pca_arch_training_grid_png(
    panel_data: dict[tuple[str, str], dict],
    *,
    regime: str,
    k_planted: int,
    graph_idx: int,
    out_path: Path,
) -> None:
    save_pca_arch_training_grid_png(
        panel_data,
        arch_order=ARCH_PLOT_ORDER,
        training_order=TRAINING_PLOT_ORDER,
        training_label_fn=_training_plot_label,
        regime=regime,
        k_planted=k_planted,
        graph_idx=graph_idx,
        n_total=TEST_N,
        out_path=out_path,
    )


def plot_pca_arch_training_grid(
    panel_data: dict[tuple[str, str], dict],
    *,
    regime: str,
    k_planted: int,
    graph_idx: int,
    out_path: Path,
) -> None:
    save_pca_arch_training_grid_html(
        panel_data,
        arch_order=ARCH_PLOT_ORDER,
        training_order=TRAINING_PLOT_ORDER,
        training_label_fn=_training_plot_label,
        regime=regime,
        k_planted=k_planted,
        graph_idx=graph_idx,
        n_total=TEST_N,
        out_path=out_path,
    )


def run_pca_embedding_plots(
    trained: list[tuple[TrainSpec, ResidualGNN]],
    device: torch.device,
) -> list[Path]:
    """PCA plots for every (architecture, training) checkpoint × eval regime."""
    if not RUN_PCA_PLOTS:
        return []

    spec_by_key: dict[tuple[str, str], tuple[TrainSpec, ResidualGNN]] = {}
    for spec, model in trained:
        spec_by_key[(spec.architecture, spec.training_way)] = (spec, model)

    saved: list[Path] = []
    for regime in tqdm(REGIME_PRINT_ORDER, desc="PCA grids", unit="regime"):
        k_min, k_max = REGIMES[regime]
        seed = pca_plot_seed(regime, PCA_GRAPH_IDX)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        k_planted = random.randint(k_min, k_max)
        A, _planted = make_planted_clique(TEST_N, k_planted, device)

        panel: dict[tuple[str, str], dict] = {}
        arch_tasks = [
            (arch, tw)
            for arch in ARCH_PLOT_ORDER
            for tw in TRAINING_PLOT_ORDER
            if tw in TRAINING_BY_ARCH.get(arch, ())
        ]
        for arch, tw in tqdm(arch_tasks, desc=f"  {regime} panels", leave=False, unit="panel"):
            entry = spec_by_key.get((arch, tw))
            if entry is None:
                continue
            spec, model = entry
            model.eval()
            data = gnn_hidden_pca_full_graph(model, A)
            panel[(arch, tw)] = data

            tag = spec.name.replace("+", "_")
            single = PCA_PLOT_DIR / f"{tag}_{regime}_g{PCA_GRAPH_IDX}_pca.html"
            plot_pca_single_html(
                data,
                title=f"{arch} · {_training_plot_label(tw)} ({regime})",
                out_path=single,
            )
            saved.append(single)

        grid_stem = f"grid_{regime}_g{PCA_GRAPH_IDX}_SGNN_SGU_SGGU_x_objective_PR"
        grid_html = PCA_PLOT_DIR / f"{grid_stem}.html"
        plot_pca_arch_training_grid(
            panel,
            regime=regime,
            k_planted=k_planted,
            graph_idx=PCA_GRAPH_IDX,
            out_path=grid_html,
        )
        saved.append(grid_html)

        if PCA_SAVE_PNG:
            grid_png = PCA_PLOT_DIR / f"{grid_stem}.png"
            plot_pca_arch_training_grid_png(
                panel,
                regime=regime,
                k_planted=k_planted,
                graph_idx=PCA_GRAPH_IDX,
                out_path=grid_png,
            )
            saved.append(grid_png)

    return saved


# ============================================================
# Train / evaluate
# ============================================================


def build_train_specs() -> list[TrainSpec]:
    specs: list[TrainSpec] = []
    for arch, updater_type, use_grad, neighbor_gate in ARCHITECTURES:
        for training_way in TRAINING_BY_ARCH[arch]:
            safe_arch = arch.replace("+", "_").replace("-", "_")
            specs.append(TrainSpec(
                name=f"{safe_arch}__{training_way}_train",
                architecture=arch,
                updater_type=updater_type,
                use_gradient_in_updater=use_grad,
                neighbor_gate_updater=neighbor_gate,
                training_way=training_way,
            ))
    return specs


def build_model(spec: TrainSpec, device: torch.device) -> ResidualGNN:
    return ResidualGNN(
        hidden=HIDDEN,
        layers=INTERNAL_STEPS,
        updater_type=spec.updater_type,
        use_gradient_in_updater=spec.use_gradient_in_updater,
        neighbor_gate_updater=spec.neighbor_gate_updater,
    ).to(device)


def save_model_checkpoint(model: ResidualGNN, spec: TrainSpec) -> None:
    if not SAVE_OUTPUTS:
        return
    path = MODEL_DIR / f"{spec.name}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "spec": asdict(spec),
        "hidden": HIDDEN,
        "layers": INTERNAL_STEPS,
        "kless": True,
        "run_name": RUN_NAME,
        "note": "No true k used in model/loss/GU/decoder. k only for data generation and evaluation.",
    }, path)


def _remap_checkpoint_keys(state_dict: dict) -> dict:
    # "update" was renamed to "gnn" in the message-passing block.
    return {
        ("gnn." + k[len("update."):] if k.startswith("update.") else k): v
        for k, v in state_dict.items()
    }


def load_model_checkpoint(spec: TrainSpec, device: torch.device) -> ResidualGNN:
    path = MODEL_DIR / f"{spec.name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}")
    checkpoint = torch.load(path, map_location=device)
    model = build_model(spec, device)
    model.load_state_dict(_remap_checkpoint_keys(checkpoint["model_state_dict"]))
    print(f"Loaded checkpoint: {path}")
    return model


def train_model(spec: TrainSpec, device: torch.device) -> ResidualGNN:
    model = build_model(spec, device)
    gul_lr = 1e-4  # GUL linear updater is fragile — lower LR than default 1e-3
    opt = torch.optim.AdamW(model.parameters(), lr=gul_lr if spec.updater_type == "GUL" else LR)

    print(f"\nTraining {spec.name}")
    print("=" * (9 + len(spec.name)))

    use_ba_mix = spec.training_way.startswith("ba_")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses: list[float] = []
        ba_count = 0

        for batch_idx in range(1, TRAIN_GRAPHS_PER_EPOCH + 1):
            opt.zero_grad()
            if use_ba_mix:
                A, _planted, _k, gtype = sample_ba_mixed_train_graph(device)
                if gtype == "BA":
                    ba_count += 1
            else:
                A, _planted, _k = sample_train_graph(device)
            loss = train_loss(model, A, spec.training_way)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
            if batch_idx % 100 == 0 or batch_idx == TRAIN_GRAPHS_PER_EPOCH:
                print(f"  epoch={epoch:02d}/{EPOCHS} batch={batch_idx:04d}/{TRAIN_GRAPHS_PER_EPOCH} loss={loss.item():.4f}", flush=True)

        mean_loss = sum(losses) / len(losses)
        if use_ba_mix:
            ba_frac = ba_count / len(losses)
            print(f"epoch={epoch:04d}/{EPOCHS} loss={mean_loss:.6f} ba={ba_frac:.0%}", flush=True)
        else:
            print(f"epoch={epoch:04d}/{EPOCHS} loss={mean_loss:.6f}", flush=True)

    return model


def spec_row_metadata(spec: TrainSpec) -> dict:
    return {
        "model": spec.name,
        "architecture": spec.architecture,
        "updater_type": spec.updater_type,
        "use_gradient_in_updater": spec.use_gradient_in_updater,
        "neighbor_gate_updater": spec.neighbor_gate_updater,
        "training_way": spec.training_way,
    }


@torch.no_grad()
def evaluate_model(model: ResidualGNN, spec: TrainSpec, device: torch.device) -> tuple[dict, list[dict]]:
    model.eval()
    decoders = DECODERS_BY_ARCH[spec.architecture]
    out: dict[str, dict] = {}
    rows: list[dict] = []

    for regime, (k_min, k_max) in REGIMES.items():
        out[regime] = {}

        for decoder in decoders:
            metric_lists: dict[str, list[float]] = {
                "approx": [], "overlap": [], "planted_exact": [],
                "clique_at_least_k": [], "found_size": [], "is_clique": [], "seconds": [],
                "stopping_steps": [],
            }

            for graph_idx in range(EVAL_GRAPHS_PER_REGIME):
                k_for_generation_only = random.randint(k_min, k_max)
                instance_seed = random.randint(0, 2**31 - 1)

                py_state = random.getstate()
                torch_state = torch.random.get_rng_state()
                cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

                random.seed(instance_seed)
                torch.manual_seed(instance_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(instance_seed)

                if PRINT_EVAL_PROGRESS:
                    print(
                        f"[eval] {spec.name} regime={regime} decoder={decoder} "
                        f"graph={graph_idx + 1}/{EVAL_GRAPHS_PER_REGIME} "
                        f"k_for_eval_only={k_for_generation_only}",
                        flush=True,
                    )

                A, planted = make_planted_clique(TEST_N, k_for_generation_only, device)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                metrics = eval_one_decoder(model, A, planted, decoder)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                metrics["seconds"] = time.perf_counter() - t0

                random.setstate(py_state)
                torch.random.set_rng_state(torch_state)
                if torch.cuda.is_available() and cuda_state is not None:
                    torch.cuda.set_rng_state_all(cuda_state)

                rows.append({
                    **spec_row_metadata(spec),
                    "regime": regime,
                    "decoder": decoder,
                    "graph_idx": graph_idx,
                    "instance_seed": instance_seed,
                    "n": TEST_N,
                    "k_for_generation_and_eval_only": k_for_generation_only,
                    **metrics,
                })

                for key, value in metrics.items():
                    metric_lists[key].append(float(value))

            for metric_name, vals in metric_lists.items():
                m, s = _mean_std_list(vals)
                out[regime].setdefault(decoder, {})[metric_name] = m
                out[regime][decoder][f"{metric_name}_std"] = s

    return out, rows


def _get_metric(all_results: dict, model_name: str, regime: str, decoder: str, metric: str) -> str:
    try:
        mean = all_results[model_name][regime][decoder][metric]
        std = all_results[model_name][regime][decoder].get(f"{metric}_std", 0.0)
        return _fmt_pm(mean, std, decimals=3)
    except KeyError:
        return "  -- "


def print_compact_results(all_results: dict[str, dict], metric: str = "approx") -> None:
    regimes_short = [("easy", "E"), ("medium", "M"), ("hard", "H")]
    col_w = 12
    model_col = 35

    for training_way in ["objective", "self_predictor_lpr"]:
        model_names = [name for name in all_results if f"__{training_way}_train" in name]
        if not model_names:
            continue

        print(f"\n{'=' * 160}")
        print(
            f"TRAINING: {training_way} / metric={metric} "
            f"(mean±std over n={EVAL_GRAPHS_PER_REGIME} graphs per regime)"
        )
        print(f"{'=' * 160}")

        header = f"{'model':<{model_col}}"
        for decoder in ALL_DECODERS:
            header += " | " + f"{decoder:<{3 * col_w + 6}}"
        print(header)

        sub = " " * model_col
        for _decoder in ALL_DECODERS:
            sub += " | " + " | ".join(f"{short:>{col_w}}" for _regime, short in regimes_short)
        print(sub)
        print("-" * len(header))

        for model_name in model_names:
            row = f"{model_name:<{model_col}}"
            for decoder in ALL_DECODERS:
                vals = " | ".join(
                    _get_metric(all_results, model_name, regime, decoder, metric)
                    for regime, _short in regimes_short
                )
                row += " | " + vals
            print(row)


def assert_kless_runtime_source() -> None:
    forbidden_runtime_names = ["CachedCliqueGradientState", "clique_objective_loss"]
    for name in forbidden_runtime_names:
        if name in globals():
            raise RuntimeError(f"Forbidden k-dependent runtime object still exists: {name}")


# ============================================================
# Main
# ============================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper train/eval: SGNN / SGNN+U / SGNN+GU (Sec. 3.4).")
    p.add_argument(
        "--train-regime",
        choices=list(REGIMES),
        default=TRAIN_REGIME,
        help="Planted-k range used during training (Table 2 uses medium-trained).",
    )
    p.add_argument(
        "--skip-training",
        action="store_true",
        help="Load checkpoints and evaluate only.",
    )
    p.add_argument(
        "--run-degree-corr",
        action="store_true",
        help="Also run inline UPR–degree correlation (slow; prefer: experiments degree).",
    )
    p.add_argument(
        "--pca-plots",
        action="store_true",
        help="Emit PCA grid HTML under plots/pca/ (train diagnostic; paper figs: gu_plots.py).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override global random seed (default: 0). Non-default seed appends _seedN to run dir.",
    )
    p.add_argument(
        "--layers",
        type=int,
        default=None,
        help="Override GNN message-passing layers (default: 4). Non-default appends _LN to run dir.",
    )
    p.add_argument(
        "--only-arch",
        nargs="+",
        default=None,
        metavar="ARCH",
        help="Train/eval only these architectures (e.g. SGNN+GUL). Default: all.",
    )
    p.add_argument(
        "--only-training-way",
        nargs="+",
        default=None,
        metavar="WAY",
        help="Train/eval only these training ways (e.g. self_predictor_lpr). Default: all.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override EPOCHS and save to _E{n} run dir to avoid overwriting defaults.",
    )
    p.add_argument(
        "--sit-steps",
        type=int,
        default=None,
        help="Override SELF_LPR_STEPS (None = full graph n-1). Use 0 for full-graph (same as None).",
    )
    p.add_argument(
        "--eval-instances",
        type=int,
        default=None,
        help="Override EVAL_GRAPHS_PER_REGIME for a quick check.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    apply_run_config(train_regime=args.train_regime, layers=args.layers, epochs=args.epochs)

    global SKIP_TRAINING, RUN_DEGREE_CORR_CHECK, RUN_PCA_PLOTS, SEED, INTERNAL_STEPS
    global OUTPUT_DIR, MODEL_DIR, PER_INSTANCE_CSV, AGGREGATE_CSV, RUN_CONFIG_JSON
    global DEGREE_CORR_CSV, PCA_PLOT_DIR, SNAP_GT_CACHE_DIR
    global SNAP_PER_INSTANCE_CSV, SNAP_AGGREGATE_CSV, RUN_NAME
    global SELF_LPR_STEPS, EVAL_GRAPHS_PER_REGIME

    if args.skip_training:
        SKIP_TRAINING = True
    if args.sit_steps is not None:
        SELF_LPR_STEPS = None if args.sit_steps == 0 else args.sit_steps
    if args.eval_instances is not None:
        EVAL_GRAPHS_PER_REGIME = args.eval_instances
    if args.run_degree_corr:
        RUN_DEGREE_CORR_CHECK = True
    if args.pca_plots:
        RUN_PCA_PLOTS = True

    if args.seed is not None:
        SEED = args.seed
        if args.seed != 0:
            new_name = RUN_NAME + f"_seed{args.seed}"
            RUN_NAME = new_name
            OUTPUT_DIR = Path("runs") / RUN_NAME
            MODEL_DIR = OUTPUT_DIR / "models"
            PER_INSTANCE_CSV = OUTPUT_DIR / "per_instance_results.csv"
            AGGREGATE_CSV = OUTPUT_DIR / "aggregate_results.csv"
            RUN_CONFIG_JSON = OUTPUT_DIR / "run_config.json"
            DEGREE_CORR_CSV = OUTPUT_DIR / "upr_priority_degree_corr.csv"
            PCA_PLOT_DIR = OUTPUT_DIR / "plots" / "pca"
            SNAP_GT_CACHE_DIR = OUTPUT_DIR / "snap_gt_cache"
            SNAP_PER_INSTANCE_CSV = OUTPUT_DIR / "snap_per_instance_results.csv"
            SNAP_AGGREGATE_CSV = OUTPUT_DIR / "snap_aggregate_results.csv"

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    assert_kless_runtime_source()
    ensure_output_dirs()
    save_run_config()

    device = torch.device(DEVICE)
    specs = build_train_specs()
    if args.only_arch:
        specs = [s for s in specs if s.architecture in args.only_arch]
        print(f"--only-arch filter: {[s.name for s in specs]}")
    if args.only_training_way:
        specs = [s for s in specs if s.training_way in args.only_training_way]
        print(f"--only-training-way filter: {[s.name for s in specs]}")

    print(f"device={device} hidden={HIDDEN} layers={INTERNAL_STEPS}")
    print(f"train_n={TRAIN_N} test_n={TEST_N} epochs={EPOCHS}")
    print(f"architectures={[a for a, _, _, _ in ARCHITECTURES]}")
    print(f"skip_training={SKIP_TRAINING}")
    print(f"outputs={OUTPUT_DIR}")
    print("K-less guarantee: k is used only for graph generation and evaluation metrics.")
    print("Decoders: one_pass/lpr/upr are O(n^2); rerun_pruned is O(n^3).")

    trained: list[tuple[TrainSpec, ResidualGNN]] = []
    for spec in specs:
        ckpt_path = MODEL_DIR / f"{spec.name}.pt"
        if SKIP_TRAINING or ckpt_path.exists():
            print(f"Loading checkpoint: {ckpt_path}")
            model = load_model_checkpoint(spec, device)
        else:
            model = train_model(spec, device)
            save_model_checkpoint(model, spec)
        trained.append((spec, model))

    all_results: dict[str, dict] = {}
    all_rows: list[dict] = []
    all_degree_corr_rows: list[dict] = []

    for spec, model in trained:
        print(f"\n=== EVALUATING: {spec.name} ===")
        results, rows = evaluate_model(model, spec, device)
        all_results[spec.name] = results
        all_rows.extend(rows)

        if RUN_DEGREE_CORR_CHECK:
            print(f"\n=== PRIORITY–DEGREE CORR: {spec.name} ===")
            corr_rows = run_degree_corr_diagnostics(model, spec, device)
            all_degree_corr_rows.extend(corr_rows)

        write_csv(PER_INSTANCE_CSV, all_rows)
        write_csv(AGGREGATE_CSV, aggregate_rows(all_rows))

    for metric in ["approx", "overlap", "planted_exact", "clique_at_least_k", "found_size", "is_clique", "seconds", "stopping_steps"]:
        print_compact_results(all_results, metric=metric)

    if RUN_DEGREE_CORR_CHECK and all_degree_corr_rows:
        print_degree_corr_summary(all_degree_corr_rows)
        write_csv(DEGREE_CORR_CSV, all_degree_corr_rows)

    if RUN_PCA_PLOTS:
        pca_paths = run_pca_embedding_plots(trained, device)
        if pca_paths:
            grids = [p for p in pca_paths if p.name.startswith("grid_")]
            tqdm.write(f"PCA: {len(pca_paths)} files under {PCA_PLOT_DIR}")
            for g in grids:
                tqdm.write(f"  grid → {g.name}")

    write_csv(PER_INSTANCE_CSV, all_rows)
    write_csv(AGGREGATE_CSV, aggregate_rows(all_rows))

    if SAVE_OUTPUTS:
        print(f"\nSaved outputs under: {OUTPUT_DIR}")
        print(f"Models: {MODEL_DIR}")
        print(f"Per-instance CSV: {PER_INSTANCE_CSV}")
        print(f"Aggregate CSV: {AGGREGATE_CSV}")
        if RUN_DEGREE_CORR_CHECK and all_degree_corr_rows:
            print(f"Priority–degree correlations: {DEGREE_CORR_CSV}")
        if RUN_PCA_PLOTS:
            print(f"PCA plots: {PCA_PLOT_DIR}")


if __name__ == "__main__":
    main()