#!/usr/bin/env python3
"""
All plotting for the k-less SGNN / SGNN+U / SGNN+GU paper stack.

User guide: see ``README.md`` in the repo root (or ``python -m src.gu_plots --help``).

What this module provides
-------------------------
1. **Paper Fig. 2–3** (CLI) — UPR diagnostics for SGNN+U and SGNN+GU (SIT checkpoints):
   PCA of GNN hidden states; removal score vs degree at UPR steps 0, 1, 5, 10, 300.
   Export: Plotly HTML (``--format html``) and/or matplotlib PDF (``--format paper``).

2. **Train PCA grids** (import only) — ``save_pca_arch_training_grid_html/png`` used when the
   main trainer is run with ``--pca-plots`` (all architectures × training objectives).

3. **Toolkit** (import only) — generic Plotly scatter helpers for representation dumps.

Entry point
-----------
From the repository root::

    python -m src.gu_plots --train-regime medium              # HTML + PDF (default)
    python -m src.gu_plots --format html --graphs 1          # interactive only
    python -m src.gu_plots --format paper --extra-pca-all-arch

Outputs live under ``runs/sgnn_paper_{regime}/plots/`` (HTML) and ``.../plots/paper/`` (PDF).
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

Format = Literal["html", "paper", "both"]

# Eval band for figure labels (keep aligned with experiments.py degree section).
REGIME = "medium"
K_MIN = 36
K_MAX = 61
DEGREE_PERCENTILE_SLIP_THRESHOLD = 15

# Filled by configure_paths() before any figure run.
PLOT_DIR: Path = Path("runs/sgnn_paper_medium/plots")
PAPER_DIR: Path = PLOT_DIR / "paper"
SEED = 0
TEST_N = 1000

# ---------------------------------------------------------------------------
# Defaults (override via CLI where noted)
# ---------------------------------------------------------------------------

HTML_GRAPH_INDICES: tuple[int, ...] = (1,)
PAPER_GRAPH_INDICES: tuple[int, ...] = (0, 1)
SNAPSHOT_STEPS: tuple[int, ...] = (0, 1, 5, 10, 300)
MAX_UPR_STEP = max(SNAPSHOT_STEPS)


def instance_seed(graph_idx: int) -> int:
    return SEED + (hash(REGIME) % 1_000) + graph_idx * 97


def _train_module():
    from . import train as m

    return m


def _run_specs():
    m = _train_module()
    return (
        m.TrainSpec(
            name="SGNN_GU__self_predictor_lpr_train",
            architecture="SGNN+GU",
            updater_type="GU",
            use_gradient_in_updater=True,
            neighbor_gate_updater=True,
            training_way="self_predictor_lpr",
        ),
        m.TrainSpec(
            name="SGNN_U__self_predictor_lpr_train",
            architecture="SGNN+U",
            updater_type="U",
            use_gradient_in_updater=False,
            neighbor_gate_updater=True,
            training_way="self_predictor_lpr",
        ),
    )


def configure_paths(*, output_dir: Path, seed: int, test_n: int) -> None:
    global PLOT_DIR, PAPER_DIR, SEED, TEST_N
    SEED = seed
    TEST_N = test_n
    PLOT_DIR = output_dir / "plots"
    PAPER_DIR = PLOT_DIR / "paper"

# Compact Plotly (train / generic scatter)
_COMPACT_WIDTH = 960
_COMPACT_HEIGHT = 720
DEFAULT_MAX_POINTS = 1000
DEFAULT_SEED = 0

ScalarStates = dict[str, tuple[np.ndarray, np.ndarray]]

# Large Plotly (interactive paper figures)
_FIG_WIDTH = 2600
_FIG_HEIGHT = 1800
PLOT_FONT_SIZE = 34
PLOT_TITLE_SIZE = 44
PLOT_AXIS_TITLE_SIZE = 40
PLOT_TICK_SIZE = 32
PLOT_LEGEND_SIZE = 38
PLOT_MARKER_SCATTER = 18
PLOT_MARKER_ARGMIN = 40
PLOT_MARKER_DEG = 36
PLOT_LINE_WIDTH = 6

# Matplotlib publication
FIG_SINGLE_SIZE = (26.0, 22.0)
SAVE_DPI = 450
FONT_BASE = 64
FONT_LABEL = 78
FONT_TICK = 60
FONT_LEGEND = 70
FONT_UPR_LEGEND = 84
FONT_COLORBAR = FONT_LEGEND
RC_PARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": FONT_BASE,
    "axes.labelsize": FONT_LABEL,
    "axes.titlesize": FONT_LABEL,
    "xtick.labelsize": FONT_TICK,
    "ytick.labelsize": FONT_TICK,
    "legend.fontsize": FONT_LEGEND,
    "axes.linewidth": 3.2,
    "xtick.major.width": 2.6,
    "ytick.major.width": 2.6,
    "lines.linewidth": 6.5,
    "savefig.dpi": SAVE_DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
COLOR_SCATTER = "#3d5a80"
COLOR_P15 = "#9b2226"
COLOR_ARGMIN_SCORE = "#bb3e03"
ALPHA_SCATTER = 0.65
SCATTER_SIZE = 360
SCATTER_SIZE_PCA = 420
MARKER_ARGMIN_SCORE_SIZE = 72
MARKER_ARGMIN_SCORE_WIDTH = 14.0
P15_LINE_WIDTH = 9.0
LEGEND_P15_LINE_WIDTH = 2.4
LEGEND_MARKERSCALE = 3.1
LEGEND_BORDERPAD = 1.2
LEGEND_LABELSPACING = 1.1
LEGEND_HANDLELENGTH = 4.4
LEGEND_HANDLETEXTPAD = 1.45
TICK_LENGTH = 22
TICK_WIDTH = 2.8
LEGEND_MARKER_SIZE = 16
LEGEND_MARKER_SIZE_ARGMIN = 24


# =============================================================================
# Plotly toolkit (train-time scatter / PCA helpers)
# =============================================================================


def _compact_layout() -> dict:
    return {
        "width": _COMPACT_WIDTH,
        "height": _COMPACT_HEIGHT,
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "font": {"family": "Inter, system-ui, sans-serif", "color": "#222"},
        "margin": {"l": 64, "r": 48, "t": 96, "b": 64},
    }


def _figure_layout() -> dict:
    return {
        "width": _FIG_WIDTH,
        "height": _FIG_HEIGHT,
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "font": {
            "family": "Inter, system-ui, sans-serif",
            "color": "#222",
            "size": PLOT_FONT_SIZE,
        },
        "margin": {"l": 130, "r": 100, "t": 160, "b": 120},
        "legend": {
            "font": {"size": PLOT_LEGEND_SIZE},
            "itemsizing": "constant",
            "itemwidth": 70,
        },
    }


def _axis_font_kwargs() -> dict:
    return {
        "title_font": {"size": PLOT_AXIS_TITLE_SIZE},
        "tickfont": {"size": PLOT_TICK_SIZE},
    }


def subsample_indices(n: int, max_points: int, seed: int = DEFAULT_SEED) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def _axis_style(title: str) -> dict:
    return {
        "title": title,
        "gridcolor": "rgba(0,0,0,0.12)",
        "zerolinecolor": "rgba(0,0,0,0.25)",
        "linecolor": "#333",
        "tickfont": {"color": "#333"},
    }


def save_scatter_plot(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    color: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    color_label: str,
    planted: np.ndarray | None = None,
    *,
    subtitle: str = "",
    enabled: bool = True,
    max_points: int = DEFAULT_MAX_POINTS,
    seed: int = DEFAULT_SEED,
    equal_aspect: bool = False,
) -> None:
    if not enabled:
        return

    import plotly.graph_objects as go

    path = path.with_suffix(".html")
    path.parent.mkdir(parents=True, exist_ok=True)

    idx = subsample_indices(len(x), max_points, seed=seed)
    x = np.asarray(x, dtype=np.float64)[idx]
    y = np.asarray(y, dtype=np.float64)[idx]
    color = np.asarray(color, dtype=np.float64)[idx]

    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="vertices",
            marker={
                "size": 7,
                "color": color,
                "colorscale": "Turbo",
                "opacity": 0.85,
                "line": {"color": "rgba(0,0,0,0.25)", "width": 0.4},
                "colorbar": {
                    "title": {"text": color_label, "font": {"color": "#222"}},
                    "tickfont": {"color": "#444"},
                    "thickness": 16,
                    "len": 0.72,
                },
            },
            hovertemplate=(
                f"{xlabel}: %{{x:.3f}}<br>"
                f"{ylabel}: %{{y:.3f}}<br>"
                f"{color_label}: %{{marker.color}}<extra></extra>"
            ),
        )
    )

    if planted is not None:
        planted = np.asarray(planted, dtype=bool)[idx]
        planted_idx = np.where(planted)[0]
        if planted_idx.size:
            fig.add_trace(
                go.Scatter(
                    x=x[planted_idx],
                    y=y[planted_idx],
                    mode="markers",
                    name="planted",
                    marker={
                        "size": 12,
                        "color": "rgba(0,0,0,0)",
                        "line": {"color": "black", "width": 1.2},
                    },
                    hovertemplate=(
                        f"{xlabel}: %{{x:.3f}}<br>"
                        f"{ylabel}: %{{y:.3f}}<br>planted<extra></extra>"
                    ),
                )
            )

    yaxis = _axis_style(ylabel)
    if equal_aspect:
        yaxis["scaleanchor"] = "x"
        yaxis["scaleratio"] = 1

    title_text = f"<b>{title}</b>"
    if subtitle:
        title_text += f"<br><sup>{subtitle}</sup>"

    fig.update_layout(
        **_compact_layout(),
        title={
            "text": title_text,
            "x": 0.5,
            "xanchor": "center",
            "font": {"color": "#111"},
        },
        xaxis=_axis_style(xlabel),
        yaxis=yaxis,
        showlegend=bool(
            planted is not None and np.asarray(planted, dtype=bool).any()
        ),
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.5, "xanchor": "center"},
    )
    fig.write_html(path, include_plotlyjs="cdn")


def gu_plot_prefix(
    plot_dir: Path,
    model_name: str,
    regime: str,
    graph_idx: int,
    instance_seed: int,
    decoder: str | None = None,
    decoder_step: int | None = None,
) -> Path:
    safe_name = (
        str(model_name)
        .replace("+", "p")
        .replace("/", "_")
        .replace(" ", "_")
    )
    path = plot_dir / safe_name / regime / f"graph_{graph_idx:03d}_seed_{instance_seed}"
    if decoder is not None:
        path = path / decoder
    if decoder_step is not None:
        path = path / f"step_{decoder_step:02d}"
    return path


def save_scalar_state_plots(
    plot_prefix: Path,
    spec_name: str,
    regime: str,
    state_name: str,
    val_sub: np.ndarray,
    deg_sub: np.ndarray,
    pl_sub: np.ndarray,
    *,
    subtitle: str = "",
    enabled: bool = True,
    max_points: int = DEFAULT_MAX_POINTS,
    seed: int = DEFAULT_SEED,
) -> None:
    save_scatter_plot(
        path=plot_prefix / f"{state_name}__value_vs_residual_degree",
        x=deg_sub,
        y=val_sub,
        color=pl_sub,
        title=f"{spec_name} | {regime} | {state_name}: value vs residual degree",
        xlabel="residual degree",
        ylabel=state_name,
        color_label="planted",
        planted=pl_sub,
        subtitle=subtitle,
        enabled=enabled,
        max_points=max_points,
        seed=seed,
    )


def save_vector_state_plots(
    plot_prefix: Path,
    state_name: str,
    coords: np.ndarray,
    evr: np.ndarray,
    deg_sub: np.ndarray,
    pl_sub: np.ndarray,
    scalar_states: ScalarStates,
    alive_np: np.ndarray,
    *,
    pc1_degree_subtitle: str = "",
    enabled: bool = True,
    max_points: int = DEFAULT_MAX_POINTS,
    seed: int = DEFAULT_SEED,
) -> None:
    pc1 = coords[:, 0]
    pc2 = coords[:, 1]
    ev0, ev1 = float(evr[0]), float(evr[1])

    save_scatter_plot(
        path=plot_prefix / f"{state_name}__pc1_vs_residual_degree",
        x=deg_sub,
        y=pc1,
        color=pl_sub,
        title=f"{state_name}: PC1 vs residual degree | EVR=({ev0:.3f}, {ev1:.3f})",
        xlabel="residual degree",
        ylabel="PC1",
        color_label="planted",
        planted=pl_sub,
        subtitle=pc1_degree_subtitle,
        enabled=enabled,
        max_points=max_points,
        seed=seed,
    )
    save_scatter_plot(
        path=plot_prefix / f"{state_name}__pc1_pc2_colored_by_residual_degree",
        x=pc1,
        y=pc2,
        color=deg_sub,
        title=f"{state_name}: PC1/PC2 colored by residual degree | EVR=({ev0:.3f}, {ev1:.3f})",
        xlabel=f"PC1 ({ev0:.3f})",
        ylabel=f"PC2 ({ev1:.3f})",
        color_label="residual degree",
        planted=pl_sub,
        enabled=enabled,
        max_points=max_points,
        seed=seed,
        equal_aspect=True,
    )

    for scalar_name, (scalar_values, scalar_alive) in scalar_states.items():
        if not np.array_equal(scalar_alive, alive_np):
            continue
        sv_sub = np.asarray(scalar_values, dtype=np.float64)[scalar_alive.astype(bool)]
        save_scatter_plot(
            path=plot_prefix / f"{state_name}__pc1_pc2_colored_by_{scalar_name}",
            x=pc1,
            y=pc2,
            color=sv_sub,
            title=f"{state_name}: PC1/PC2 colored by {scalar_name} (residual subgraph)",
            xlabel=f"PC1 ({ev0:.3f})",
            ylabel=f"PC2 ({ev1:.3f})",
            color_label=scalar_name,
            planted=pl_sub,
            enabled=enabled,
            max_points=max_points,
            seed=seed,
            equal_aspect=True,
        )


def save_pca_arch_training_grid_html(
    panel_data: dict[tuple[str, str], dict],
    *,
    arch_order: tuple[str, ...],
    training_order: tuple[str, ...],
    training_label_fn,
    regime: str,
    k_planted: int,
    graph_idx: int,
    n_total: int,
    out_path: Path,
) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    nrows, ncols = len(arch_order), len(training_order)
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[
            f"{arch} · {training_label_fn(tw)}"
            for arch in arch_order
            for tw in training_order
        ],
        horizontal_spacing=0.06,
        vertical_spacing=0.10,
    )

    for row, arch in enumerate(arch_order, start=1):
        for col, tw in enumerate(training_order, start=1):
            data = panel_data.get((arch, tw))
            if data is None:
                fig.add_trace(
                    go.Scatter(
                        x=[0.0],
                        y=[0.0],
                        mode="text",
                        text=["not trained"],
                        textfont=dict(size=14, color="#888"),
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )
                fig.update_xaxes(visible=False, row=row, col=col)
                fig.update_yaxes(visible=False, row=row, col=col)
                continue

            marker: dict = {
                "size": 6,
                "color": data["degree"],
                "colorscale": "Turbo",
                "opacity": 0.85,
                "line": {"color": "rgba(0,0,0,0.2)", "width": 0.35},
            }
            if row == 1 and col == ncols:
                marker["colorbar"] = {
                    "title": {"text": "Degree", "font": {"size": 11}},
                    "tickfont": {"size": 10},
                    "thickness": 14,
                    "len": 0.72,
                }

            fig.add_trace(
                go.Scatter(
                    x=data["pc1"],
                    y=data["pc2"],
                    mode="markers",
                    marker=marker,
                    hovertemplate="PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>degree: %{marker.color}<extra></extra>",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            pc1_var = 100.0 * float(data["pc1_var"])
            pc2_var = 100.0 * float(data["pc2_var"])
            rho = float(data["spearman_pc1_degree"])
            fig.update_xaxes(title_text=f"PC1 ({pc1_var:.0f}%)", row=row, col=col)
            fig.update_yaxes(title_text=f"PC2 ({pc2_var:.0f}%) · ρ={rho:+.2f}", row=row, col=col)

    fig.update_layout(
        template="plotly_white",
        title={
            "text": (
                "<b>PCA — SGNN / SGNN+U / SGNN+GU × objective / PR</b><br>"
                f"<sup>full graph n={n_total}, eval regime={regime}, graph={graph_idx}, "
                f"k={k_planted}; one GNN pass, PCA(hidden), color=degree</sup>"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        width=320 * ncols + 120,
        height=280 * nrows + 140,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=11, color="#222"),
        showlegend=False,
        margin=dict(l=48, r=48, t=100, b=48),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def save_pca_arch_training_grid_png(
    panel_data: dict[tuple[str, str], dict],
    *,
    arch_order: tuple[str, ...],
    training_order: tuple[str, ...],
    training_label_fn,
    regime: str,
    k_planted: int,
    graph_idx: int,
    n_total: int,
    out_path: Path,
) -> None:
    nrows, ncols = len(arch_order), len(training_order)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), dpi=150, layout="constrained"
    )
    if nrows == 1:
        axes = np.asarray([axes])
    if ncols == 1:
        axes = axes.reshape(-1, 1)

    last_sc = None
    for row, arch in enumerate(arch_order):
        for col, tw in enumerate(training_order):
            ax = axes[row, col]
            label = f"{arch} · {training_label_fn(tw)}"
            data = panel_data.get((arch, tw))
            if data is None:
                ax.text(0.5, 0.5, "not trained", ha="center", va="center", fontsize=11, color="#888")
                ax.set_title(label, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_facecolor("#f5f5f5")
                continue

            degree = data["degree"]
            norm = mcolors.Normalize(vmin=float(degree.min()), vmax=float(degree.max()))
            last_sc = ax.scatter(
                data["pc1"],
                data["pc2"],
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
            ax.set_title(label, fontsize=10)
            ax.set_facecolor("white")
            ax.grid(True, color=(0, 0, 0, 0.1), linewidth=0.5)

    if last_sc is not None:
        fig.colorbar(last_sc, ax=axes[:, -1].tolist(), shrink=0.85, label="Degree")

    fig.suptitle(
        f"PCA of GNN hidden — full graph n={n_total}  |  regime={regime}  k={k_planted}  graph={graph_idx}",
        fontsize=12,
    )
    fig.patch.set_facecolor("white")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# UPR data (shared by HTML + paper exporters)
# =============================================================================


@torch.no_grad()
def gnn_hidden_pca_kless(model, A: torch.Tensor, alive: torch.Tensor) -> dict:
    m = _train_module()
    h, _logits, _probs = model.hidden_logits_probs(A, alive)
    h_np = h.detach().cpu().numpy()
    degree = m.residual_degree_numpy(A, alive)
    alive_np = alive.detach().cpu().numpy().astype(bool)
    h_alive = h_np[alive_np]
    deg_alive = degree[alive_np]

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(h_alive)
    pc1, pc2 = pcs[:, 0], pcs[:, 1]

    if h_alive.shape[0] >= 2 and np.std(pc1) > 1e-12:
        sp = float(np.corrcoef(pc1, deg_alive)[0, 1])
        if sp < 0:
            pc1 = -pc1
            pcs[:, 0] = pc1

    return {
        "pc1": pc1,
        "pc2": pcs[:, 1],
        "degree": deg_alive,
        "pc1_var": float(pca.explained_variance_ratio_[0]),
        "pc2_var": float(pca.explained_variance_ratio_[1]),
    }


@torch.no_grad()
def collect_upr_snapshots(
    model,
    A: torch.Tensor,
    *,
    max_step: int,
    snapshot_steps: tuple[int, ...],
) -> dict[int, dict]:
    m = _train_module()
    n = A.shape[0]
    alive = torch.ones(n, dtype=torch.bool, device=A.device)
    deg_alive, alive_edges = m.init_alive_degrees_and_edges(A)
    alive_count = n

    scores = model.normal_scores(A, alive)
    use_grad = bool(model.use_gradient_in_updater)
    grad_cache = (
        m.CachedKlessCliqueGradientState.build(A, scores, alive) if use_grad else None
    )

    want = set(snapshot_steps)
    snapshots: dict[int, dict] = {}

    for step in range(max_step + 1):
        if step in want:
            alive_np = alive.detach().cpu().numpy().astype(bool)
            scores_np = scores.detach().cpu().numpy()
            deg_np = deg_alive.detach().cpu().numpy()
            alive_idx = np.where(alive_np)[0]
            scores_sub = scores_np[alive_np]
            deg_sub = deg_np[alive_np]
            argmin_score = int(m.remove_lowest(scores, alive).item())
            argmin_deg = int(alive_idx[int(np.argmin(deg_sub))])
            idx_score = int(np.argmin(scores_sub))
            p15_degree = float(np.percentile(deg_sub, DEGREE_PERCENTILE_SLIP_THRESHOLD))
            snapshots[step] = {
                "alive_count": int(alive_count),
                "scores": scores_sub.astype(np.float64),
                "degree": deg_sub.astype(np.float64),
                "p15_degree": p15_degree,
                "argmin_score": argmin_score,
                "argmin_degree": argmin_deg,
                "same_argmin": int(argmin_score == argmin_deg),
                "argmin_score_pct_weak": float(
                    100.0 * np.mean(deg_sub <= deg_sub[idx_score])
                ),
            }

        if step >= max_step:
            break
        if alive_count <= 1 or m.alive_is_clique_fast(alive_count, alive_edges):
            break

        scores_before = scores.clone()
        grad = grad_cache.gradient() if grad_cache is not None else None
        v = m.remove_lowest(scores, alive)

        alive, deg_alive, alive_edges = m.remove_vertex_update_state(
            A, alive, deg_alive, alive_edges, v
        )
        alive_count -= 1
        if grad_cache is not None:
            grad_cache.delete_vertex_only(v)

        scores = model.update_scores(
            scores=scores_before,
            A=A,
            alive_after=alive,
            removed=v,
            grad=grad if use_grad else None,
        )

    return snapshots


@dataclass(frozen=True)
class GraphBundle:
    A: torch.Tensor
    alive0: torch.Tensor
    k_planted: int
    seed: int


def load_graph_bundle(graph_idx: int, device: torch.device) -> GraphBundle:
    seed = instance_seed(graph_idx)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    m = _train_module()
    k_planted = random.randint(K_MIN, K_MAX)
    A, _planted = m.make_planted_clique(TEST_N, k_planted, device)
    alive0 = torch.ones(A.shape[0], dtype=torch.bool, device=A.device)
    return GraphBundle(A=A, alive0=alive0, k_planted=k_planted, seed=seed)


# =============================================================================
# Interactive Plotly (Fig. 2–3 HTML)
# =============================================================================


def write_pc1_pc2_html(
    data: dict,
    *,
    model_name: str,
    regime: str,
    k_planted: int,
    graph_idx: int,
    out_path: Path,
) -> None:
    import plotly.graph_objects as go

    pc1_var = 100.0 * float(data["pc1_var"])
    pc2_var = 100.0 * float(data["pc2_var"])
    fig = go.Figure(
        go.Scatter(
            x=data["pc1"],
            y=data["pc2"],
            mode="markers",
            marker={
                "size": PLOT_MARKER_SCATTER + 4,
                "color": data["degree"],
                "colorscale": "Turbo",
                "opacity": 0.85,
                "line": {"color": "rgba(0,0,0,0.25)", "width": 0.6},
                "colorbar": {
                    "title": {"text": "Residual degree", "font": {"size": PLOT_AXIS_TITLE_SIZE}},
                    "tickfont": {"size": PLOT_TICK_SIZE},
                    "len": 0.75,
                    "thickness": 48,
                },
            },
            hovertemplate="PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>degree: %{marker.color}<extra></extra>",
        )
    )
    fig.update_layout(
        **_figure_layout(),
        title={
            "text": (
                f"<b>{model_name} — final GNN hidden (not logits)</b><br>"
                f"<sup>{regime}, graph {graph_idx}, k={k_planted}, n={TEST_N}; "
                f"PC1={pc1_var:.1f}%, PC2={pc2_var:.1f}% var.</sup>"
            ),
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": PLOT_TITLE_SIZE},
        },
        xaxis={"title": f"PC1 ({pc1_var:.1f}% var.)", **_axis_font_kwargs()},
        yaxis={"title": f"PC2 ({pc2_var:.1f}% var.)", **_axis_font_kwargs()},
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")


def write_upr_panels_html(
    snapshots: dict[int, dict],
    *,
    model_name: str,
    regime: str,
    k_planted: int,
    graph_idx: int,
    out_path: Path,
    snapshot_steps: tuple[int, ...] = SNAPSHOT_STEPS,
) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    steps = [s for s in snapshot_steps if s in snapshots]
    ncols = len(steps)
    fig = make_subplots(
        rows=1,
        cols=ncols,
        subplot_titles=[f"step {s} (n={snapshots[s]['alive_count']})" for s in steps],
        horizontal_spacing=0.08,
    )
    for col in range(1, ncols + 1):
        fig.layout.annotations[col - 1].font.size = PLOT_AXIS_TITLE_SIZE

    for col, step in enumerate(steps, start=1):
        snap = snapshots[step]
        scores = snap["scores"]
        degree = snap["degree"]
        p15 = snap["p15_degree"]
        idx_score = int(np.argmin(scores))
        idx_deg = int(np.argmin(degree))

        fig.add_trace(
            go.Scatter(
                x=scores,
                y=degree,
                mode="markers",
                marker={"size": PLOT_MARKER_SCATTER, "color": "rgba(60,90,140,0.45)"},
                showlegend=False,
                hovertemplate="score: %{x:.3f}<br>degree: %{y}<extra></extra>",
            ),
            row=1,
            col=col,
        )
        x_min, x_max = float(np.min(scores)), float(np.max(scores))
        pad = 0.02 * (x_max - x_min + 1e-6)
        fig.add_trace(
            go.Scatter(
                x=[x_min - pad, x_max + pad],
                y=[p15, p15],
                mode="lines",
                line={"color": "#c44e52", "width": PLOT_LINE_WIDTH, "dash": "dash"},
                showlegend=(col == 1),
                name=f"{DEGREE_PERCENTILE_SLIP_THRESHOLD}th %ile degree",
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=[scores[idx_score]],
                y=[degree[idx_score]],
                mode="markers",
                marker={
                    "symbol": "x",
                    "size": PLOT_MARKER_ARGMIN,
                    "color": "#d62728",
                    "line": {"width": 3, "color": "#d62728"},
                },
                showlegend=(col == 1),
                name="argmin score",
            ),
            row=1,
            col=col,
        )
        if snap["same_argmin"] == 0:
            fig.add_trace(
                go.Scatter(
                    x=[scores[idx_deg]],
                    y=[degree[idx_deg]],
                    mode="markers",
                    marker={
                        "symbol": "circle-open",
                        "size": PLOT_MARKER_DEG,
                        "color": "#2ca02c",
                        "line": {"width": 3},
                    },
                    showlegend=(col == 1),
                    name="argmin degree",
                ),
                row=1,
                col=col,
            )

        fig.update_xaxes(title_text="Score", row=1, col=col, **_axis_font_kwargs())
        fig.update_yaxes(title_text="Degree", row=1, col=col, **_axis_font_kwargs())

    same_note = ", ".join(
        f"s{s}:argmin={'same' if snapshots[s]['same_argmin'] else 'diff'}, "
        f"p%={snapshots[s]['argmin_score_pct_weak']:.1f}"
        for s in steps
    )
    layout = _figure_layout()
    layout.pop("width", None)
    layout.pop("height", None)
    legend = layout.pop("legend", {})
    legend.update(
        {
            "orientation": "h",
            "y": 1.14,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": PLOT_LEGEND_SIZE},
            "itemsizing": "constant",
            "itemwidth": 72,
        }
    )
    fig.update_layout(
        **layout,
        width=520 * ncols,
        height=820,
        title={
            "text": (
                f"<b>{model_name} — UPR removal score vs degree</b><br>"
                f"<sup>{regime}, graph {graph_idx}, k={k_planted}; "
                f"argmin(score) vs argmin(degree): {same_note}</sup>"
            ),
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": PLOT_TITLE_SIZE},
        },
        legend=legend,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")


def run_interactive_html(
    *,
    device: torch.device,
    graph_indices: tuple[int, ...],
    specs: tuple | None = None,
) -> None:
    m = _train_module()
    if specs is None:
        specs = _run_specs()
    print(f"HTML figures: regime={REGIME} graphs={graph_indices}")
    print(f"Output: {PLOT_DIR}")

    for graph_idx in graph_indices:
        bundle = load_graph_bundle(graph_idx, device)
        print(
            f"\n--- graph {graph_idx} k={bundle.k_planted} seed={bundle.seed} ---"
        )

        for spec in specs:
            model = m.load_model_checkpoint(spec, device)
            model.eval()
            tag = spec.name.replace("__", "_")

            pca_data = gnn_hidden_pca_kless(model, bundle.A, bundle.alive0)
            pca_path = PLOT_DIR / f"{tag}_{REGIME}_g{graph_idx}_pc1_pc2_degree.html"
            write_pc1_pc2_html(
                pca_data,
                model_name=spec.architecture,
                regime=REGIME,
                k_planted=bundle.k_planted,
                graph_idx=graph_idx,
                out_path=pca_path,
            )
            print(f"  {pca_path}")

            snapshots = collect_upr_snapshots(
                model,
                bundle.A,
                max_step=MAX_UPR_STEP,
                snapshot_steps=SNAPSHOT_STEPS,
            )
            upr_path = PLOT_DIR / f"{tag}_{REGIME}_g{graph_idx}_upr_score_vs_degree.html"
            write_upr_panels_html(
                snapshots,
                model_name=spec.architecture,
                regime=REGIME,
                k_planted=bundle.k_planted,
                graph_idx=graph_idx,
                out_path=upr_path,
            )
            print(f"  {upr_path}")
            for s in SNAPSHOT_STEPS:
                if s in snapshots:
                    snap = snapshots[s]
                    print(
                        f"    step {s}: n={snap['alive_count']} "
                        f"argmin_score==argmin_degree={bool(snap['same_argmin'])}"
                    )


# =============================================================================
# Publication matplotlib (Fig. 2–3 PDF)
# =============================================================================


def apply_paper_style() -> None:
    plt.rcParams.update(RC_PARAMS)


def _save_paper_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def write_pc1_pc2_pdf(data: dict, *, out_stem: Path) -> None:
    pc1 = np.asarray(data["pc1"], dtype=np.float64)
    pc2 = np.asarray(data["pc2"], dtype=np.float64)
    degree = np.asarray(data["degree"], dtype=np.float64)
    pc1_var_pct = 100.0 * float(data.get("pc1_var", 0.0))
    pc2_var_pct = 100.0 * float(data.get("pc2_var", 0.0))

    fig, ax = plt.subplots(figsize=FIG_SINGLE_SIZE, constrained_layout=True)
    norm = mcolors.Normalize(vmin=float(degree.min()), vmax=float(degree.max()))
    sc = ax.scatter(
        pc1,
        pc2,
        c=degree,
        cmap="cividis",
        s=SCATTER_SIZE_PCA,
        alpha=0.88,
        linewidths=0.15,
        edgecolors="white",
        norm=norm,
        rasterized=True,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Degree", fontsize=FONT_COLORBAR)
    cbar.ax.tick_params(
        axis="y",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
        labelright=False,
        labeltop=False,
        length=0,
        width=0,
    )
    cbar.set_ticks([])
    cbar.outline.set_linewidth(1.6)

    ax.set_xlabel(f"PC1 ({pc1_var_pct:.1f}%)", fontsize=FONT_LABEL)
    ax.set_ylabel(f"PC2 ({pc2_var_pct:.1f}%)", fontsize=FONT_LABEL)
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
        labelright=False,
        labeltop=False,
        length=0,
        width=0,
    )
    ax.grid(True, linestyle="-", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_paper_figure(fig, out_stem)


def _upr_legend_handles() -> tuple[list[Line2D], list[str]]:
    handles: list[Line2D] = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=LEGEND_MARKER_SIZE,
            markerfacecolor=COLOR_SCATTER,
            markeredgecolor=COLOR_SCATTER,
            alpha=0.6,
            label="Alive vertices",
        ),
        Line2D(
            [0],
            [0],
            linestyle=(0, (5, 3)),
            color=COLOR_P15,
            linewidth=LEGEND_P15_LINE_WIDTH,
            label=f"{DEGREE_PERCENTILE_SLIP_THRESHOLD}th percentile degree",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="",
            markersize=LEGEND_MARKER_SIZE_ARGMIN,
            color=COLOR_ARGMIN_SCORE,
            markeredgewidth=4.2,
            label=r"$\arg\min$ score",
        ),
    ]
    labels = [
        "Alive vertices",
        f"{DEGREE_PERCENTILE_SLIP_THRESHOLD}th percentile degree",
        r"$\arg\min$ score",
    ]
    return handles, labels


def write_upr_step_pdf(
    snap: dict,
    *,
    out_stem: Path,
    show_legend: bool = False,
) -> None:
    scores = snap["scores"]
    degree = snap["degree"]
    p15 = snap["p15_degree"]
    idx_score = int(np.argmin(scores))

    fig, ax = plt.subplots(figsize=FIG_SINGLE_SIZE, constrained_layout=True)
    ax.scatter(
        scores,
        degree,
        s=SCATTER_SIZE,
        c=COLOR_SCATTER,
        alpha=ALPHA_SCATTER,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    x_lo, x_hi = float(np.min(scores)), float(np.max(scores))
    pad = 0.03 * (x_hi - x_lo + 1e-6)
    ax.plot(
        [x_lo - pad, x_hi + pad],
        [p15, p15],
        linestyle=(0, (5, 3)),
        color=COLOR_P15,
        linewidth=P15_LINE_WIDTH,
        zorder=2,
    )
    ax.plot(
        [scores[idx_score]],
        [degree[idx_score]],
        linestyle="none",
        marker="X",
        markersize=MARKER_ARGMIN_SCORE_SIZE,
        markeredgewidth=MARKER_ARGMIN_SCORE_WIDTH,
        color=COLOR_ARGMIN_SCORE,
        zorder=4,
    )

    ax.set_xlabel("Score", fontsize=FONT_LABEL)
    ax.set_ylabel("Degree", fontsize=FONT_LABEL)
    ax.grid(True, linestyle="-", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_legend:
        handles, labels = _upr_legend_handles()
        ax.legend(
            handles,
            labels,
            loc="upper right",
            frameon=True,
            framealpha=0.92,
            fontsize=FONT_UPR_LEGEND,
            markerscale=LEGEND_MARKERSCALE,
            borderpad=LEGEND_BORDERPAD,
            labelspacing=LEGEND_LABELSPACING,
            handlelength=LEGEND_HANDLELENGTH,
            handletextpad=LEGEND_HANDLETEXTPAD,
        )
    ax.tick_params(
        axis="both",
        which="major",
        width=TICK_WIDTH,
        length=TICK_LENGTH,
        labelsize=FONT_TICK,
    )
    _save_paper_figure(fig, out_stem)


def run_publication_pdf(
    *,
    device: torch.device,
    graph_indices: tuple[int, ...],
    specs: tuple | None = None,
    extra_all_arch: bool = False,
) -> None:
    m = _train_module()
    if specs is None:
        specs = _run_specs()
    apply_paper_style()
    print(f"Paper PDFs: regime={REGIME} graphs={graph_indices}")
    print(f"Output: {PAPER_DIR}")

    all_specs: list | None = None
    if extra_all_arch:
        all_specs = [
            s
            for s in m.build_train_specs()
            if s.training_way in ("objective", "self_predictor_lpr")
        ]

    for graph_idx in graph_indices:
        bundle = load_graph_bundle(graph_idx, device)
        print(
            f"\n--- graph {graph_idx} k={bundle.k_planted} seed={bundle.seed} ---"
        )

        for spec in specs:
            model = m.load_model_checkpoint(spec, device)
            model.eval()
            tag = spec.name.replace("__", "_")

            pca_data = gnn_hidden_pca_kless(model, bundle.A, bundle.alive0)
            pca_stem = PAPER_DIR / f"{tag}_{REGIME}_g{graph_idx}_pc1_pc2_degree"
            write_pc1_pc2_pdf(pca_data, out_stem=pca_stem)
            print(f"  {pca_stem}.pdf")

            snapshots = collect_upr_snapshots(
                model,
                bundle.A,
                max_step=MAX_UPR_STEP,
                snapshot_steps=SNAPSHOT_STEPS,
            )
            for step in SNAPSHOT_STEPS:
                if step not in snapshots:
                    continue
                upr_stem = (
                    PAPER_DIR
                    / f"{tag}_{REGIME}_g{graph_idx}_upr_step{step}_score_vs_degree"
                )
                write_upr_step_pdf(
                    snapshots[step],
                    out_stem=upr_stem,
                    show_legend=(spec.architecture == "SGNN+U" and step == 0),
                )
                print(f"  {upr_stem}.pdf")

        if all_specs is not None:
            primary = {s.name for s in specs}
            for spec in all_specs:
                if spec.name in primary:
                    continue
                model = m.load_model_checkpoint(spec, device)
                model.eval()
                tag = spec.name.replace("__", "_")
                pca_data = gnn_hidden_pca_kless(model, bundle.A, bundle.alive0)
                pca_stem = PAPER_DIR / f"{tag}_{REGIME}_g{graph_idx}_pc1_pc2_degree"
                write_pc1_pc2_pdf(pca_data, out_stem=pca_stem)
                print(f"  [all-arch] {pca_stem}.pdf")


# =============================================================================
# CLI
# =============================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from . import train as exp

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--train-regime",
        choices=list(exp.REGIMES),
        default="medium",
        help="Checkpoint directory runs/sgnn_paper_{regime}/",
    )
    p.add_argument(
        "--format",
        choices=("html", "paper", "both"),
        default="both",
        help="html=Plotly under plots/; paper=PDF under plots/paper/; both=run each pipeline.",
    )
    p.add_argument(
        "--graphs",
        type=int,
        nargs="+",
        default=None,
        metavar="IDX",
        help=(
            "Graph indices to plot. Default: html→%s, paper→%s, both uses those per format."
            % (list(HTML_GRAPH_INDICES), list(PAPER_GRAPH_INDICES))
        ),
    )
    p.add_argument(
        "--extra-pca-all-arch",
        action="store_true",
        help="Paper export: also PCA PDF for every trained architecture/objective checkpoint.",
    )
    return p.parse_args(argv)


def _resolve_graph_indices(
    fmt: Format,
    graphs: list[int] | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if graphs is not None:
        idx = tuple(graphs)
        return idx, idx
    if fmt == "html":
        return HTML_GRAPH_INDICES, ()
    if fmt == "paper":
        return (), PAPER_GRAPH_INDICES
    return HTML_GRAPH_INDICES, PAPER_GRAPH_INDICES


def main(argv: list[str] | None = None) -> None:
    from . import train as exp

    args = parse_args(argv)
    exp.apply_run_config(train_regime=args.train_regime)
    configure_paths(output_dir=exp.OUTPUT_DIR, seed=exp.SEED, test_n=exp.TEST_N)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    fmt: Format = args.format
    html_idx, paper_idx = _resolve_graph_indices(fmt, args.graphs)

    if fmt in ("html", "both") and html_idx:
        run_interactive_html(device=device, graph_indices=html_idx)
    if fmt in ("paper", "both") and paper_idx:
        run_publication_pdf(
            device=device,
            graph_indices=paper_idx,
            extra_all_arch=args.extra_pca_all_arch,
        )


if __name__ == "__main__":
    main()
