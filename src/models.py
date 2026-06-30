"""
K-less residual GNN for planted-clique search.

**SGNN** — one forward pass; fixed removal scores; no UPR.
**SGNN+U** — scores updated after each removal (MLP on neighbors); UPR; no objective gradient in MLP.
**SGNN+GU** — same as +U but MLP also sees cached clique-objective gradient.
**SGNN+GUL** — GU variant with a linear updater (LayerNorm-free, column-std over alive).
**SGNN+GUC** — GU variant with an additional maintained residual aggregation channel from
               an intermediate GNN layer, projected to 4 dims; total input to MLP = 13.

All: input = alive mask only; no oracle k in forward/loss/updater.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Layer index (0-based) to freeze for GUC residual aggregation.
# Layer 1 of 4 = second message-passing step; avoids the degree-collapsed final layer.
GUC_AGG_LAYER = 1
GUC_PROJ_DIM = 4   # d→k projection; keeps U_θ input constant-size


def _std_alive(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Zero-mean unit-var standardization over alive vertices only."""
    xa = x[m]
    return (x - xa.mean()) / (xa.std() + 1e-6)


def remove_lowest(scores: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
    return scores.masked_fill(~alive.bool(), 1e9).argmin()


def masked_removal_ce(
    removal_scores: torch.Tensor, alive: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    logits = (-removal_scores).masked_fill(~alive.bool(), -1e9)
    return F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))


def clique_loss_from_probs_kless(p: torch.Tensor, A: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
    alive_f = alive.float()
    p = p * alive_f
    A_res = A * alive_f[:, None] * alive_f[None, :]
    B = (1.0 - A).clone()
    B.fill_diagonal_(0.0)
    B_res = B * alive_f[:, None] * alive_f[None, :]
    return -(p @ A_res @ p) + (p @ B_res @ p)


def clique_objective_loss_kless(scores: torch.Tensor, A: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(scores) * alive.float()
    return clique_loss_from_probs_kless(p, A, alive)


@dataclass
class CachedKlessCliqueGradientState:
    """Incremental ∂(clique objective)/∂scores for SGNN+GU (O(n) per deletion)."""

    A: torch.Tensor
    B: torch.Tensor
    alive: torch.Tensor
    p: torch.Tensor
    Ap: torch.Tensor
    Bp: torch.Tensor
    sum_p: torch.Tensor
    n: int

    @staticmethod
    def build(A: torch.Tensor, scores: torch.Tensor, alive: torch.Tensor) -> CachedKlessCliqueGradientState:
        alive_f = alive.float()
        with torch.no_grad():
            p = torch.sigmoid(scores.detach()) * alive_f
            B = (1.0 - A).clone()
            B.fill_diagonal_(0.0)
            A_res = A * alive_f[:, None] * alive_f[None, :]
            B_res = B * alive_f[:, None] * alive_f[None, :]
            return CachedKlessCliqueGradientState(
                A=A,
                B=B,
                alive=alive.clone(),
                p=p.clone(),
                Ap=A_res @ p,
                Bp=B_res @ p,
                sum_p=p.sum(),
                n=A.shape[0],
            )

    @torch.no_grad()
    def gradient(self) -> torch.Tensor:
        return (-2.0 * self.Ap + 2.0 * self.Bp) * self.alive.float()

    @torch.no_grad()
    def gradient_dgu(self) -> torch.Tensor:
        """Like gradient() but divided by Z_t = sum of alive sigmoid scores (DGU renormalization)."""
        return self.gradient() / (self.sum_p + 1e-8)

    @torch.no_grad()
    def delete_vertex_only(self, v: torch.Tensor) -> None:
        v_int = int(v.item())
        if not bool(self.alive[v_int].item()):
            return
        old_p_v = self.p[v_int].clone()
        self.Ap = self.Ap - self.A[:, v_int] * old_p_v
        self.Bp = self.Bp - self.B[:, v_int] * old_p_v
        self.sum_p = self.sum_p - old_p_v
        self.p[v_int] = 0.0
        self.alive[v_int] = False
        self.Ap[v_int] = 0.0
        self.Bp[v_int] = 0.0


@dataclass
class CachedAggregationState:
    """
    Maintained residual aggregation of a frozen intermediate GNN embedding h^(ℓ).

    Agg[i] = sum_{j alive} A[i,j] * H[j]    (O(n·d) downdate per deletion, d=64 constant)

    On deleting v: Agg -= A[:,v] * H[v]  — same column-subtraction structure as the
    gradient cache. H is frozen at build time (detached); this is exact, not an approx.
    """
    A: torch.Tensor       # (n, n)
    H: torch.Tensor       # (n, d) frozen h^(ℓ), zeroed out as vertices are removed
    alive: torch.Tensor   # (n,) bool
    Agg: torch.Tensor     # (n, d) maintained aggregation
    n: int

    @staticmethod
    def build(A: torch.Tensor, H: torch.Tensor, alive: torch.Tensor) -> CachedAggregationState:
        af = alive.float()
        H_masked = (H * af[:, None]).clone()
        A_res = A * af[:, None] * af[None, :]
        return CachedAggregationState(
            A=A,
            H=H_masked,
            alive=alive.clone(),
            Agg=(A_res @ H_masked).clone(),
            n=A.shape[0],
        )

    @torch.no_grad()
    def channel(self) -> torch.Tensor:
        """σ(Agg / (n−1)), zeroed for dead vertices. Shape (n, d)."""
        z = torch.sigmoid(self.Agg / max(self.n - 1, 1))
        return z * self.alive.float()[:, None]

    @torch.no_grad()
    def delete_vertex_only(self, v: torch.Tensor) -> None:
        vi = int(v.item())
        if not bool(self.alive[vi].item()):
            return
        hv = self.H[vi].clone()                              # (d,)
        self.Agg -= self.A[:, vi][:, None] * hv[None, :]    # O(n·d) column subtraction
        self.H[vi] = 0.0
        self.alive[vi] = False
        self.Agg[~self.alive] = 0.0                          # re-zero all dead rows

    @torch.no_grad()
    def verify(self, H_frozen: torch.Tensor, tol: float = 1e-4) -> float:
        """Recompute Agg from scratch and return max absolute error (correctness gate)."""
        af = self.alive.float()
        A_res = self.A * af[:, None] * af[None, :]
        ref = A_res @ (H_frozen * af[:, None])
        err = (self.Agg - ref).abs().max().item()
        assert err < tol, f"CachedAggregationState drifted: max_err={err:.2e} > tol={tol:.2e}"
        return err


@dataclass
class CachedFGUState:
    """
    First-order structural gradient for SGNN+FGU.

    Caches at decode start:
      H[l]        ∈ (n, d): forward activations h^l at GNN layer l
      BackGrad[l] ∈ (n, d): ∂scores_i/∂msg^l_i  (frozen; not updated after deletions)

    Per deletion of vertex v:
      fgu_signal_i = -(A[i,v]/(n-1)) * Σ_l  BackGrad[l][i] · H[l][v]

    First-order Taylor approx of Δscores when v is removed.
    O(n·d·L) per deletion = O(n²) total with d, L constant.
    """
    H: list             # L tensors, each (n, d) — zeroed as vertices are deleted
    BackGrad: list      # L tensors, each (n, d) — frozen at decode start
    alive: torch.Tensor
    n: int
    L: int
    d: int

    @staticmethod
    def build(model: "ResidualGNN", A: torch.Tensor, alive: torch.Tensor) -> "CachedFGUState":
        n = A.shape[0]
        d = model.hidden
        L = model.layers
        alive_f = alive.float().unsqueeze(-1)
        A_res = A * alive.float()[None, :]

        W1 = model.gnn[0].weight   # (d, 2d)
        W2 = model.gnn[2].weight   # (d, d)

        H_list: list[torch.Tensor] = []
        pre1_list: list[torch.Tensor] = []
        pre2_list: list[torch.Tensor] = []

        with torch.no_grad():
            h = model.input(alive_f) * alive_f
            for _ in range(L):
                H_list.append(h.clone())
                msg = (A_res @ (h * alive_f)) / max(n - 1, 1)
                cat_hm = torch.cat([h, msg], dim=-1)
                pre1 = cat_hm @ W1.T + model.gnn[0].bias
                act1 = torch.tanh(pre1)
                pre2 = act1 @ W2.T + model.gnn[2].bias
                h = torch.tanh(pre2) * alive_f
                pre1_list.append(pre1.clone())
                pre2_list.append(pre2.clone())

            # Backpropagate: ∂scores_i/∂msg^l_i for all l, i
            G = model.out.weight.expand(n, -1).clone()   # (n, d)
            BackGrad: list[torch.Tensor] = [torch.empty(0)] * L

            for l in range(L - 1, -1, -1):
                D2 = 1.0 - torch.tanh(pre2_list[l]) ** 2   # sech²(pre2) (n, d)
                D1 = 1.0 - torch.tanh(pre1_list[l]) ** 2   # sech²(pre1) (n, d)
                G_D2 = G * D2
                G_W2 = G_D2 @ W2         # (n, d)
                G_D1 = G_W2 * D1
                G_full = G_D1 @ W1       # (n, 2d)
                BackGrad[l] = G_full[:, d:].clone()   # msg part
                G = G_full[:, :d]                      # h part

        return CachedFGUState(H=H_list, BackGrad=BackGrad, alive=alive.clone(), n=n, L=L, d=d)

    @torch.no_grad()
    def fgu_signal(self, v: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Δscores_i ≈ -(A[i,v]/(n-1)) * Σ_l BackGrad[l][i]·H[l][v]  for all i."""
        vi = int(v.item())
        col_v = A[:, vi]
        inner = self.BackGrad[0].new_zeros(self.n)
        for l in range(self.L):
            inner = inner + (self.BackGrad[l] * self.H[l][vi]).sum(dim=1)
        return -(col_v / max(self.n - 1, 1)) * inner

    @torch.no_grad()
    def delete_vertex_only(self, v: torch.Tensor) -> None:
        vi = int(v.item())
        if not bool(self.alive[vi].item()):
            return
        for h in self.H:
            h[vi] = 0.0
        self.alive[vi] = False


class ResidualGNN(nn.Module):
    """
    SGNN / SGNN+U / SGNN+GU / SGNN+GUL / SGNN+GUC / SGNN+FGU / SGNN+DGU.
    ``updater_type``: ``none`` | ``U`` | ``GU`` | ``GUL`` | ``GUC`` | ``FGU`` | ``DGU``

    DGU = Dynamic GU: same 6-input MLP as GU, but the gradient signal is pre-divided
    by Z_t = sum_{j in S_t} p_j before each update step (dynamic renormalization).

    ``forward(A, alive)`` → removal logits on the residual subgraph.
    ``update_scores`` — used by LPR/UPR after a vertex is removed (+U / +GU only).
    ``embed_with_intermediate`` — used by GUC at decode-start to capture h^(GUC_AGG_LAYER).
    """

    def __init__(
        self,
        hidden: int,
        layers: int,
        updater_type: str,
        use_gradient_in_updater: bool,
        neighbor_gate_updater: bool,
        updater_hidden: int = 64,
    ):
        super().__init__()
        self.hidden = hidden
        self.layers = layers
        self.updater_type = updater_type
        self.use_gradient_in_updater = use_gradient_in_updater
        self.neighbor_gate_updater = neighbor_gate_updater

        self.input = nn.Linear(1, hidden)
        self.out = nn.Linear(hidden, 1)
        self.gnn = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        if updater_type == "none":
            self.score_updater = None
        elif updater_type == "GUL":
            # 5 features (no alive_f); column-wise std over alive handled in update_scores
            self.score_updater = nn.Linear(5, 1, bias=False)
            nn.init.normal_(self.score_updater.weight, std=1e-3)  # start ~no-op
        elif updater_type == "GUC":
            # Project frozen intermediate embedding d→k before feeding to MLP
            self.agg_proj = nn.Linear(hidden, GUC_PROJ_DIM, bias=False)
            # Input: 5 base features + z_i (k) + z_v (k) = 5 + 2*GUC_PROJ_DIM
            guc_in = 5 + 2 * GUC_PROJ_DIM
            self.score_updater = nn.Sequential(
                nn.Linear(guc_in, updater_hidden),
                nn.Tanh(),
                nn.Linear(updater_hidden, updater_hidden),
                nn.Tanh(),
                nn.Linear(updater_hidden, 1),
            )
        elif updater_type == "FGU":
            # 6 GU features + 1 FGU structural gradient signal (linearized GNN prediction)
            self.score_updater = nn.Sequential(
                nn.Linear(7, updater_hidden),
                nn.Tanh(),
                nn.Linear(updater_hidden, updater_hidden),
                nn.Tanh(),
                nn.Linear(updater_hidden, 1),
            )
        else:
            self.score_updater = nn.Sequential(
                nn.Linear(6, updater_hidden),
                nn.Tanh(),
                nn.Linear(updater_hidden, updater_hidden),
                nn.Tanh(),
                nn.Linear(updater_hidden, 1),
            )

    def _embed(self, A: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
        alive_f = alive.float().unsqueeze(-1)
        h = self.input(alive_f) * alive_f
        n = A.shape[0]
        A_res = A * alive.float()[None, :]
        for _ in range(self.layers):
            msg = (A_res @ (h * alive_f)) / max(n - 1, 1)
            h = self.gnn(torch.cat([h, msg], dim=-1)) * alive_f
        return h

    def embed_with_intermediate(
        self, A: torch.Tensor, alive: torch.Tensor, capture_layer: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run GNN and also return the frozen embedding at `capture_layer` (0-indexed).
        Used by GUC to build CachedAggregationState at decode start."""
        alive_f = alive.float().unsqueeze(-1)
        h = self.input(alive_f) * alive_f
        n = A.shape[0]
        A_res = A * alive.float()[None, :]
        captured = None
        for li in range(self.layers):
            msg = (A_res @ (h * alive_f)) / max(n - 1, 1)
            h = self.gnn(torch.cat([h, msg], dim=-1)) * alive_f
            if li == capture_layer:
                captured = h.detach().clone()
        assert captured is not None, f"capture_layer={capture_layer} out of range for layers={self.layers}"
        return h, captured

    def forward(self, A: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
        """GNN → one logit per vertex (masked dead = -inf)."""
        logits = self.out(self._embed(A, alive)).squeeze(-1)
        return logits.masked_fill(~alive.bool(), -1e9)

    def hidden_states(self, A: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
        return self._embed(A, alive)

    def hidden_logits_probs(
        self, A: torch.Tensor, alive: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self._embed(A, alive)
        logits = self.out(h).squeeze(-1).masked_fill(~alive.bool(), -1e9)
        probs = torch.sigmoid(logits).masked_fill(~alive.bool(), 0.0)
        return h, logits, probs

    def normal_scores(self, A: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
        return self.forward(A, alive)

    def update_scores(
        self,
        scores: torch.Tensor,
        A: torch.Tensor,
        alive_after: torch.Tensor,
        removed: torch.Tensor,
        grad: torch.Tensor | None,
        agg_channel: torch.Tensor | None = None,
        fgu_vec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.score_updater is None:
            return scores.masked_fill(~alive_after.bool(), -1e9)

        m = alive_after.bool()
        n = A.shape[0]
        v = int(removed.item())
        alive_f = alive_after.float()
        affected = A[:, v]
        if self.use_gradient_in_updater:
            if grad is None:
                raise RuntimeError("SGNN+GU/GUL/GUC needs gradient cache")
            grad_i, grad_v = grad, grad[v].expand(n)
        else:
            grad_i = grad_v = torch.zeros_like(scores)

        if self.updater_type == "GUL":
            feats = torch.stack([
                _std_alive(scores, m),
                scores[v].expand(n),
                affected,
                _std_alive(grad_i, m),
                _std_alive(grad_v, m),
            ], dim=1)  # (n, 5)

            delta = self.score_updater(feats).squeeze(-1)
            delta = delta - delta[m].mean()   # shift-invariant: kills accumulating bias
            delta = delta * alive_f           # mask dead; no neighbor gate for GUL

            scores = scores + delta
            sa = scores[m]
            scores = (scores - sa.mean()) / (sa.std() + 1e-6)  # renormalize each step
            return scores.masked_fill(~m, -1e9)

        if self.updater_type == "GUC":
            if agg_channel is None:
                raise RuntimeError("GUC update_scores needs agg_channel from CachedAggregationState")
            z_proj = self.agg_proj(agg_channel)          # (n, GUC_PROJ_DIM)
            z_i = z_proj                                  # per-vertex channel
            z_v = z_proj[v].unsqueeze(0).expand(n, -1)   # removed vertex's channel, broadcast
            feats = torch.cat([
                torch.stack([scores, scores[v].expand(n), affected, grad_i, grad_v], dim=1),  # (n, 5)
                z_i,   # (n, GUC_PROJ_DIM)
                z_v,   # (n, GUC_PROJ_DIM)
            ], dim=1)  # (n, 5 + 2*GUC_PROJ_DIM = 13)

            delta = self.score_updater(feats).squeeze(-1)
            if self.neighbor_gate_updater:
                delta = delta * affected * alive_f
            else:
                delta = delta * alive_f
            return (scores + delta).masked_fill(~m, -1e9)

        if self.updater_type == "FGU":
            if fgu_vec is None:
                raise RuntimeError("FGU needs fgu_vec from CachedFGUState.fgu_signal()")
            feats = torch.stack(
                [scores, scores[v].expand(n), affected, grad_i, grad_v, alive_f, fgu_vec],
                dim=1,
            )  # (n, 7)
            delta = self.score_updater(feats).squeeze(-1)
            if self.neighbor_gate_updater:
                delta = delta * affected * alive_f
            else:
                delta = delta * alive_f
            return (scores + delta).masked_fill(~m, -1e9)

        # GU / U path
        delta = self.score_updater(
            torch.stack(
                [scores, scores[v].expand(n), affected, grad_i, grad_v, alive_f],
                dim=1,
            )
        ).squeeze(-1)

        if self.neighbor_gate_updater:
            delta = delta * affected * alive_f
        else:
            delta = delta * alive_f

        return (scores + delta).masked_fill(~m, -1e9)


@dataclass
class TrainSpec:
    name: str
    architecture: str
    updater_type: str
    use_gradient_in_updater: bool
    neighbor_gate_updater: bool
    training_way: str
