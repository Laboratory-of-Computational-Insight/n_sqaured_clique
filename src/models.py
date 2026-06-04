"""
K-less residual GNN for planted-clique search.

**SGNN** — one forward pass; fixed removal scores; no UPR.
**SGNN+U** — scores updated after each removal (MLP on neighbors); UPR; no objective gradient in MLP.
**SGNN+GU** — same as +U but MLP also sees cached clique-objective gradient.

All three: input = alive mask only; no oracle k in forward/loss/updater.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ResidualGNN(nn.Module):
    """
    SGNN / SGNN+U / SGNN+GU (``updater_type``: ``none`` | ``U`` | ``GU``).

    ``forward(A, alive)`` → removal logits on the residual subgraph.
    ``update_scores`` — used by LPR/UPR after a vertex is removed (+U / +GU only).
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
    ) -> torch.Tensor:
        if self.score_updater is None:
            return scores.masked_fill(~alive_after.bool(), -1e9)

        n = A.shape[0]
        v = int(removed.item())
        alive_f = alive_after.float()
        affected = A[:, v]
        if self.use_gradient_in_updater:
            if grad is None:
                raise RuntimeError("SGNN+GU needs gradient cache")
            grad_i, grad_v = grad, grad[v].expand(n)
        else:
            grad_i = grad_v = torch.zeros_like(scores)

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

        return (scores + delta).masked_fill(~alive_after.bool(), -1e9)


@dataclass
class TrainSpec:
    name: str
    architecture: str
    updater_type: str
    use_gradient_in_updater: bool
    neighbor_gate_updater: bool
    training_way: str
