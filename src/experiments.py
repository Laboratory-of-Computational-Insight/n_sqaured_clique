#!/usr/bin/env python3
"""
Post-training experiments (everything except train + planted-clique eval).

Use after ``python -m src.train`` has written checkpoints under
``runs/sgnn_paper_{regime}/models/``.

Subcommands
-----------
snap     Table 6 — zero-shot U-LPR on SNAP graphs (cached max-clique labels).
bench    Table 3 / Appendix K — wall-clock decoding on random G(n, ½).
degree   Tables 4–5 — UPR removal vs degree / gradient statistics.

Examples
--------
::

    python -m src.experiments --help
    python -m src.experiments snap --train-regime medium
    python -m src.experiments bench --ns 1000 --instances 100
    python -m src.experiments degree --train-regime medium

See ``README.md`` in the repo root for setup, outputs, and prerequisites.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

from . import train
from .models import CachedKlessCliqueGradientState, TrainSpec, remove_lowest

# ---------------------------------------------------------------------------
# SNAP (Table 6)
# ---------------------------------------------------------------------------

SNAP_DATASETS = (
    "twitter",
    "collab",
    "imdb_binary",
    "com-youtube",
    "com-orkut",
    "facebook",
)

SNAP_SPECS = (
    TrainSpec(
        name="SGNN__objective_train",
        architecture="SGNN",
        updater_type="none",
        use_gradient_in_updater=False,
        neighbor_gate_updater=False,
        training_way="objective",
    ),
    TrainSpec(
        name="SGNN_U__self_predictor_lpr_train",
        architecture="SGNN+U",
        updater_type="U",
        use_gradient_in_updater=False,
        neighbor_gate_updater=True,
        training_way="self_predictor_lpr",
    ),
    TrainSpec(
        name="SGNN_GU__self_predictor_lpr_train",
        architecture="SGNN+GU",
        updater_type="GU",
        use_gradient_in_updater=True,
        neighbor_gate_updater=True,
        training_way="self_predictor_lpr",
    ),
    TrainSpec(
        name="SGNN_DGU__self_predictor_lpr_train",
        architecture="SGNN+DGU",
        updater_type="DGU",
        use_gradient_in_updater=True,
        neighbor_gate_updater=True,
        training_way="self_predictor_lpr",
    ),
)


def _snap_cache_path(dataset_name: str) -> Path:
    return train.SNAP_GT_CACHE_DIR / f"{dataset_name}.jsonl"


def _load_snap_gt_cache(dataset_name: str) -> dict[int, dict]:
    path = _snap_cache_path(dataset_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing ground-truth cache: {path}\n"
            "Expected snap_gt_cache/*.jsonl under the train-regime run directory."
        )

    cached: dict[int, dict] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            cached[int(rec["graph_idx"])] = rec

    dataset_path = Path("datasets") / f"{dataset_name}.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    with dataset_path.open() as f:
        n_graphs = sum(1 for line in f if line.strip())

    if len(cached) != n_graphs:
        raise ValueError(
            f"{dataset_name}: cache has {len(cached)} graphs, dataset has {n_graphs}"
        )

    print(f"[snap] {dataset_name}: {len(cached)} graphs", flush=True)
    return cached


def _planted_from_snap_cache(record: dict, device: torch.device) -> torch.Tensor:
    n = int(record["n"])
    planted = torch.zeros(n, dtype=torch.bool, device=device)
    planted[torch.tensor(record["vertices"], dtype=torch.long, device=device)] = True
    return planted


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _ldr_row_metadata() -> dict:
    return {
        "model": "LDR",
        "architecture": "LDR",
        "updater_type": "none",
        "use_gradient_in_updater": False,
        "neighbor_gate_updater": False,
        "training_way": "none",
    }


@torch.no_grad()
def _evaluate_snap(
    model: train.ResidualGNN,
    spec: TrainSpec,
    device: torch.device,
    *,
    datasets: tuple[str, ...],
    decoders: tuple[str, ...],
) -> list[dict]:
    model.eval()
    rows: list[dict] = []

    for dataset_name in datasets:
        dataset_path = Path("datasets") / f"{dataset_name}.jsonl"
        if not dataset_path.exists():
            print(f"SKIP {dataset_name}: missing {dataset_path}", flush=True)
            continue

        cache = _load_snap_gt_cache(dataset_name)
        with dataset_path.open() as f:
            lines = f.readlines()

        print(f"\n=== {spec.name} on {dataset_name} ({len(lines)} graphs) ===", flush=True)

        for graph_idx, line in enumerate(tqdm(lines, desc=f"snap:{dataset_name}", unit="graph")):
            rec = cache[graph_idx]
            row = json.loads(line)
            adj = np.array(row["adjacency_matrix"], dtype=np.float32)
            n = adj.shape[0]
            if int(rec["n"]) != n:
                raise ValueError(
                    f"{dataset_name} graph {graph_idx}: cache n={rec['n']} != adj n={n}"
                )

            A = torch.tensor(adj, dtype=torch.float32, device=device)
            planted = _planted_from_snap_cache(rec, device)
            max_k = int(rec["max_clique_size"])

            for decoder in decoders:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                metrics = train.eval_one_decoder(model, A, planted, decoder)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                metrics["seconds"] = time.perf_counter() - t0

                rows.append({
                    **train.spec_row_metadata(spec),
                    "eval_split": "snap",
                    "regime": dataset_name,
                    "decoder": decoder,
                    "graph_idx": graph_idx,
                    "n": n,
                    "gt_source": rec.get("source", "cache"),
                    "k_for_generation_and_eval_only": max_k,
                    **metrics,
                })

    return rows


@torch.no_grad()
def _evaluate_snap_ldr(
    device: torch.device,
    *,
    datasets: tuple[str, ...],
) -> list[dict]:
    rows: list[dict] = []
    for dataset_name in datasets:
        dataset_path = Path("datasets") / f"{dataset_name}.jsonl"
        if not dataset_path.exists():
            print(f"SKIP {dataset_name} (LDR): missing {dataset_path}", flush=True)
            continue

        cache = _load_snap_gt_cache(dataset_name)
        with dataset_path.open() as f:
            lines = f.readlines()

        print(f"\n=== LDR on {dataset_name} ({len(lines)} graphs) ===", flush=True)

        for graph_idx, line in enumerate(tqdm(lines, desc=f"ldr:{dataset_name}", unit="graph")):
            rec = cache[graph_idx]
            row = json.loads(line)
            adj = np.array(row["adjacency_matrix"], dtype=np.float32)
            n = adj.shape[0]

            A = torch.tensor(adj, dtype=torch.float32, device=device)
            planted = _planted_from_snap_cache(rec, device)
            max_k = int(rec["max_clique_size"])

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            metrics = train.eval_one_decoder(None, A, planted, "ldr")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            metrics["seconds"] = time.perf_counter() - t0

            rows.append({
                **_ldr_row_metadata(),
                "eval_split": "snap",
                "regime": dataset_name,
                "decoder": "ldr",
                "graph_idx": graph_idx,
                "n": n,
                "gt_source": rec.get("source", "cache"),
                "k_for_generation_and_eval_only": max_k,
                **metrics,
            })

    return rows


def _print_snap_summary(rows: list[dict]) -> None:
    agg = train.aggregate_rows(rows)
    print("\n=== SNAP approx (Table 6 style) ===", flush=True)
    for row in sorted(agg, key=lambda r: (r["model"], r["regime"], r["decoder"])):
        print(
            f"  {row['model']:<40} {row['regime']:<14} {row['decoder']:<14} "
            f"approx={row['approx_mean']:.3f}±{row['approx_std']:.3f}  n={row['num_instances']}",
            flush=True,
        )


def cmd_snap(args: argparse.Namespace) -> None:
    train.apply_run_config(
        train_regime=args.train_regime,
        layers=getattr(args, "layers", None),
        epochs=getattr(args, "epochs", None),
    )
    train.SNAP_GT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(train.SEED)
    torch.manual_seed(train.SEED)
    device = torch.device(train.DEVICE)

    # Separate LDR (no checkpoint) from model-based decoders
    requested = tuple(args.decoders)
    model_decoders = tuple(d for d in requested if d != "ldr")
    run_ldr = "ldr" in requested

    use_both = "both" in args.models
    specs: list[TrainSpec] = []
    if use_both or "sgnn" in args.models:
        specs.append(SNAP_SPECS[0])
    if use_both or "u" in args.models:
        specs.append(SNAP_SPECS[1])
    if use_both or "gu" in args.models:
        specs.append(SNAP_SPECS[2])
    if use_both or "dgu" in args.models:
        specs.append(SNAP_SPECS[3])

    print(
        f"snap: device={device} regime={args.train_regime} "
        f"decoders={requested} out={train.OUTPUT_DIR}",
        flush=True,
    )

    t0 = time.perf_counter()
    all_rows: list[dict] = []

    # LDR needs no model
    if run_ldr:
        all_rows.extend(_evaluate_snap_ldr(device, datasets=tuple(args.datasets)))

    # Model-based decoders — evaluate one dataset at a time, print+save after each
    if model_decoders:
        for spec in specs:
            model = train.load_model_checkpoint(spec, device)
            for dataset_name in args.datasets:
                new_rows = _evaluate_snap(
                    model, spec, device,
                    datasets=(dataset_name,), decoders=model_decoders,
                )
                all_rows.extend(new_rows)
                # Print aggregate for this dataset immediately
                agg = train.aggregate_rows(new_rows)
                for row in agg:
                    print(
                        f"[snap done: {dataset_name}] {row['model']:<40} {row['regime']:<14} "
                        f"{row['decoder']:<14} approx={row['approx_mean']:.3f}±{row['approx_std']:.3f}"
                        f"  n={row['num_instances']}",
                        flush=True,
                    )
                # Incrementally save everything collected so far
                _write_csv(train.SNAP_PER_INSTANCE_CSV, all_rows)
                _write_csv(train.SNAP_AGGREGATE_CSV, train.aggregate_rows(all_rows))

    print(f"\nSaved {train.SNAP_PER_INSTANCE_CSV}", flush=True)
    print(f"\nSaved {train.SNAP_AGGREGATE_CSV}", flush=True)
    print(f"Done in {time.perf_counter() - t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Timing benchmark (Table 3 / Appendix K)
# ---------------------------------------------------------------------------

BENCH_NS = (500, 1000, 1500, 2000, 5000)
BENCH_DECODER_KEYS = {"topk": "one_pass", "rerun": "rerun_pruned", "ulpr": "upr"}

BENCH_ARCH_CONFIG: tuple[tuple[str, TrainSpec, tuple[str, ...]], ...] = (
    (
        "sgnn",
        TrainSpec(
            name="SGNN__objective_train",
            architecture="SGNN",
            updater_type="none",
            use_gradient_in_updater=False,
            neighbor_gate_updater=False,
            training_way="objective",
        ),
        ("topk", "rerun"),
    ),
    (
        "sgnnu",
        TrainSpec(
            name="SGNN_U__self_predictor_lpr_train",
            architecture="SGNN+U",
            updater_type="U",
            use_gradient_in_updater=False,
            neighbor_gate_updater=True,
            training_way="self_predictor_lpr",
        ),
        ("topk", "rerun", "ulpr"),
    ),
    (
        "sgnngu",
        TrainSpec(
            name="SGNN_GU__self_predictor_lpr_train",
            architecture="SGNN+GU",
            updater_type="GU",
            use_gradient_in_updater=True,
            neighbor_gate_updater=True,
            training_way="self_predictor_lpr",
        ),
        ("topk", "rerun", "ulpr"),
    ),
)


def _bench_random_er(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    A = torch.bernoulli(torch.full((n, n), train.P_ER, device=device))
    A = torch.triu(A, diagonal=1)
    A = A + A.t()
    A.fill_diagonal_(0.0)
    planted = torch.zeros(n, dtype=torch.bool, device=device)
    return A, planted


def _bench_instance_seed(n: int, graph_idx: int) -> int:
    return train.SEED + n * 1_000_003 + graph_idx * 97


def _bench_mean_std(vals: list[float]) -> tuple[float, float]:
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / len(vals)
    return mean, var**0.5


@torch.no_grad()
def cmd_bench(args: argparse.Namespace) -> None:
    train.apply_run_config(train_regime=args.train_regime)

    sizes = tuple(args.ns) if args.ns is not None else BENCH_NS
    random.seed(train.SEED)
    device = torch.device(train.DEVICE)

    def log(msg: str = "") -> None:
        print(msg, flush=True)

    log(f"bench: device={device} P_ER={train.P_ER} instances={args.instances} "
        f"checkpoints={args.train_regime}")
    log("decoder labels: topk=one_pass, rerun=rerun_pruned, ulpr=upr")
    log()

    models: list[tuple[str, TrainSpec, tuple[str, ...], object]] = []
    for label, spec, decoders in BENCH_ARCH_CONFIG:
        model = train.load_model_checkpoint(spec, device)
        model.eval()
        models.append((label, spec, decoders, model))

    columns: list[tuple[str, str]] = []
    for label, _, decoders, _ in models:
        for dec in decoders:
            columns.append((label, dec))

    header = f"{'n':>6} | " + " | ".join(
        f"{arch}_{dec}"[:14].ljust(14) for arch, dec in columns
    )
    log(header)
    log("-" * len(header))

    for n in sizes:
        log(f"[n={n}] start")
        col_times: dict[tuple[str, str], list[float]] = {c: [] for c in columns}

        for graph_idx in range(args.instances):
            seed = _bench_instance_seed(n, graph_idx)
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            A, planted = _bench_random_er(n, device)

            for label, _spec, decoders, model in models:
                for dec in decoders:
                    key = BENCH_DECODER_KEYS[dec]
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    train.eval_one_decoder(model, A, planted, key)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    col_times[(label, dec)].append(time.perf_counter() - t0)

            if (graph_idx + 1) % max(1, args.instances // 5) == 0:
                log(f"  [n={n}] {graph_idx + 1}/{args.instances} graphs done")

        row = f"{n:>6} | "
        row += " | ".join(
            f"{_bench_mean_std(col_times[c])[0]:.3f}±{_bench_mean_std(col_times[c])[1]:.3f}".ljust(14)
            for c in columns
        )
        log(row)

    log("done")


# ---------------------------------------------------------------------------
# Degree analysis (Tables 4–5)
# ---------------------------------------------------------------------------

NUM_GRAPHS = 1000  # Paper Table 4–5 (medium regime)
NUM_STEPS = 500
PRINT_FULL_STEP_TABLE = NUM_STEPS <= 50
# Steps where Spearman(scores, degree) is typically weak (~0.17–0.2).
WEAK_SPEARMAN_STEP_LO = 1
WEAK_SPEARMAN_STEP_HI = 49
REGIME = "medium"
K_MIN = 36
K_MAX = 61

GU_SPEC = TrainSpec(
    name="SGNN_GU__self_predictor_lpr_train",
    architecture="SGNN+GU",
    updater_type="GU",
    use_gradient_in_updater=True,
    neighbor_gate_updater=True,
    training_way="self_predictor_lpr",
)

U_SPEC = TrainSpec(
    name="SGNN_U__self_predictor_lpr_train",
    architecture="SGNN+U",
    updater_type="U",
    use_gradient_in_updater=False,
    neighbor_gate_updater=True,
    training_way="self_predictor_lpr",
)

# Models to run (same planted graphs / seeds for fair comparison).
RUN_SPECS: tuple[TrainSpec, ...] = (GU_SPEC, U_SPEC)


# Step bands for above-median rate reporting (name, lo, hi); lo/hi None = all steps.
STEP_BANDS: tuple[tuple[str, int | None, int | None], ...] = (
    ("all", None, None),
    ("step_0", 0, 0),
    ("steps_1_49", WEAK_SPEARMAN_STEP_LO, WEAK_SPEARMAN_STEP_HI),
    ("steps_7_24", 7, 24),
)

DEGREE_PERCENTILE_SLIP_THRESHOLD = 15
PERCENTILE_THRESHOLDS = (30, 25, 20, 15, 10, 5, 2, 1)
PERCENTILE_TABLE_THRESHOLDS: tuple[int, ...] = (50,) + PERCENTILE_THRESHOLDS

ABOVE_MEDIAN_BY_GRAPH_CSV: Path
ABOVE_MEDIAN_AGGREGATE_CSV: Path
ABOVE_P15_BY_GRAPH_CSV: Path
ABOVE_P15_AGGREGATE_CSV: Path
PERCENTILE_LE_T_TABLE_CSV: Path


def output_paths_for(spec: TrainSpec) -> tuple[Path, Path, Path]:
    tag = spec.name
    root = train.OUTPUT_DIR
    return (
        root / f"gu_degree_grad_check__{tag}.csv",
        root / f"gu_removal_degree_percentile__{tag}.csv",
        root / f"gu_above_median_removals__{tag}.csv",
    )

SCORE_CANDIDATES = ("degree", "grad", "neg_grad", "Ap", "Bp", "compat")
SCORE_PROBES: dict[str, tuple[str, ...]] = {
    "degree": ("degree",),
    "grad": ("grad",),
    "degree+grad": ("degree", "grad"),
    "degree+neg_grad": ("degree", "neg_grad"),
    "degree+grad+Ap+Bp+compat": ("degree", "grad", "Ap", "Bp", "compat"),
}

DELTA_CANDIDATES = ("degree_delta", "grad_delta", "compat_delta")
DELTA_PROBES: dict[str, tuple[str, ...]] = {
    "degree_delta": ("degree_delta",),
    "grad_delta": ("grad_delta",),
    "degree_delta+grad_delta": ("degree_delta", "grad_delta"),
    "degree_delta+grad_delta+compat_delta": ("degree_delta", "grad_delta", "compat_delta"),
}


def instance_seed(graph_idx: int) -> int:
    # Same seeds for all models → identical issue graphs.
    return _DEGREE_SEED + (hash(REGIME) % 1_000) + graph_idx * 97


def _alive_sub(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)[mask.astype(bool)]


def removed_vertex_degree_stats(
    deg_alive: np.ndarray,
    alive_np: np.ndarray,
    v_int: int,
) -> dict[str, float | int]:
    """
    Degree of the removed vertex among alive nodes (before deletion).

    degree_percentile_weak: % of alive nodes with residual degree <= removed (0–100).
    degree_percentile_midrank: tie-aware rank fraction * 100.
    is_argmin_degree: 1 if removed vertex has minimum residual degree on alive set.
    is_above_median_degree: 1 if removed degree > median(alive residual degrees).
    rank_index_from_median: |removed rank index - median rank index| in alive degree order
        (0 = lowest degree); nan if not above median.
    """
    nan_idx = float("nan")
    alive_deg = _alive_sub(deg_alive, alive_np)
    d_rem = float(deg_alive[v_int])
    if alive_deg.size == 0:
        return {
            "removed_vertex": v_int,
            "removed_degree": d_rem,
            "alive_median_degree": float("nan"),
            "degree_percentile_weak": float("nan"),
            "degree_percentile_midrank": float("nan"),
            "is_min_degree": 0,
            "is_argmin_degree": 0,
            "is_above_median_degree": 0,
            "is_below_median_degree": 0,
            "is_at_or_below_median_degree": 0,
            "alive_p15_degree": float("nan"),
            "is_above_p15_degree": 0,
            "is_at_or_below_p15_degree": 0,
            "rank_index_from_median": nan_idx,
            "degree_minus_median": float("nan"),
        }

    median_deg = float(np.median(alive_deg))
    pct_weak = 100.0 * float(np.mean(alive_deg <= d_rem))
    below = float(np.sum(alive_deg < d_rem))
    equal = float(np.sum(np.isclose(alive_deg, d_rem)))
    midrank = (below + 0.5 * equal) / max(alive_deg.size, 1)
    min_deg = float(np.min(alive_deg))
    alive_idx = np.where(alive_np)[0]
    argmin_v = int(alive_idx[int(np.argmin(alive_deg))])
    is_above = int(d_rem > median_deg)
    is_below = int(d_rem < median_deg)
    is_at_or_below_median = int(d_rem <= median_deg)
    p15_deg = float(np.percentile(alive_deg, DEGREE_PERCENTILE_SLIP_THRESHOLD))
    is_above_p15 = int(pct_weak > DEGREE_PERCENTILE_SLIP_THRESHOLD)
    is_at_or_below_p15 = int(pct_weak <= DEGREE_PERCENTILE_SLIP_THRESHOLD)

    # Rank index in ascending alive-degree order (tie-aware center of tie block).
    sorted_deg = np.sort(alive_deg)
    tie_start = int(np.searchsorted(sorted_deg, d_rem, side="left"))
    tie_count = int(np.sum(np.isclose(alive_deg, d_rem)))
    rank_removed_idx = tie_start + 0.5 * max(tie_count - 1, 0)
    rank_median_idx = 0.5 * (alive_deg.size - 1)
    rank_index_from_median = abs(rank_removed_idx - rank_median_idx) if is_above else nan_idx
    degree_minus_median = (d_rem - median_deg) if is_above else float("nan")

    return {
        "removed_vertex": v_int,
        "removed_degree": d_rem,
        "alive_median_degree": median_deg,
        "degree_percentile_weak": pct_weak,
        "degree_percentile_midrank": 100.0 * midrank,
        "is_min_degree": int(d_rem <= min_deg + 1e-9),
        "is_argmin_degree": int(v_int == argmin_v),
        "is_above_median_degree": is_above,
        "is_below_median_degree": is_below,
        "is_at_or_below_median_degree": is_at_or_below_median,
        "alive_p15_degree": p15_deg,
        "is_above_p15_degree": is_above_p15,
        "is_at_or_below_p15_degree": is_at_or_below_p15,
        "rank_index_from_median": rank_index_from_median,
        "degree_minus_median": degree_minus_median,
    }


def _corr_pair(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return float("nan"), float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        # e.g. degree_delta == -1 for every neighbor of the removed vertex
        return float("nan"), float("nan")
    sp = stats.spearmanr(x, y).statistic
    pr = stats.pearsonr(x, y).statistic
    return float(sp), float(pr)


def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    std = float(x.std())
    if std < 1e-12:
        return x - float(x.mean())
    return (x - float(x.mean())) / std


def _r2_ols(y: np.ndarray, *feature_cols: np.ndarray) -> float:
    """R^2 of standardized y ~ standardized features (OLS)."""
    y = _standardize(y)
    if y.size < 2:
        return float("nan")
    if not feature_cols:
        return float("nan")
    X = np.column_stack([_standardize(f) for f in feature_cols])
    if X.shape[0] < X.shape[1] + 1:
        return float("nan")
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return float("nan")
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _append_row(
    rows: list[dict],
    *,
    graph_idx: int,
    step: int,
    k: int,
    target: str,
    candidate: str,
    mask: str,
    spearman: float = float("nan"),
    pearson: float = float("nan"),
    r2: float = float("nan"),
) -> None:
    rows.append(
        {
            "graph_idx": graph_idx,
            "step": step,
            "k": k,
            "target": target,
            "candidate": candidate,
            "mask": mask,
            "spearman": spearman,
            "pearson": pearson,
            "r2": r2,
        }
    )


def _concept_dict(
    grad_cache: CachedKlessCliqueGradientState | None,
    deg_alive: torch.Tensor,
) -> dict[str, np.ndarray]:
    degree = deg_alive.detach().cpu().numpy()
    if grad_cache is None:
        return {"degree": degree}
    grad = grad_cache.gradient().detach().cpu().numpy()
    ap = grad_cache.Ap.detach().cpu().numpy()
    bp = grad_cache.Bp.detach().cpu().numpy()
    return {
        "degree": degree,
        "grad": grad,
        "neg_grad": -grad,
        "Ap": ap,
        "Bp": bp,
        "compat": ap - bp,
    }


@torch.no_grad()
def collect_degree_grad_check(
    model,
    A: torch.Tensor,
    k_planted: int,
    *,
    graph_idx: int,
    max_steps: int,
) -> tuple[list[dict], list[dict]]:
    n = A.shape[0]
    alive = torch.ones(n, dtype=torch.bool, device=A.device)
    deg_alive, alive_edges = train.init_alive_degrees_and_edges(A)
    alive_count = n

    scores = model.normal_scores(A, alive)
    use_grad = bool(model.use_gradient_in_updater)
    grad_cache = (
        CachedKlessCliqueGradientState.build(A, scores, alive) if use_grad else None
    )

    rows: list[dict] = []
    removal_rows: list[dict] = []
    initial_degree_full: np.ndarray | None = None
    initial_median_degree = float("nan")

    for step in range(max_steps):
        if alive_count <= 1 or train.alive_is_clique_fast(alive_count, alive_edges):
            break

        alive_np = alive.detach().cpu().numpy().astype(bool)
        scores_before = scores.clone()
        scores_np = scores_before.detach().cpu().numpy()

        grad_before = grad_cache.gradient() if grad_cache is not None else None
        concepts = _concept_dict(grad_cache, deg_alive)
        scores_sub = _alive_sub(scores_np, alive_np)

        if step == 0:
            # Frozen full-graph residual degrees at UPR step 0 (all nodes alive).
            initial_degree_full = concepts["degree"].copy()
            initial_median_degree = float(np.median(_alive_sub(initial_degree_full, alive_np)))

        for cand in SCORE_CANDIDATES:
            if cand not in concepts:
                continue
            feat_sub = _alive_sub(concepts[cand], alive_np)
            if feat_sub.shape != scores_sub.shape:
                continue
            sp, pr = _corr_pair(scores_sub, feat_sub)
            _append_row(
                rows,
                graph_idx=graph_idx,
                step=step,
                k=k_planted,
                target="scores",
                candidate=cand,
                mask="alive",
                spearman=sp,
                pearson=pr,
            )

        for probe_name, cols in SCORE_PROBES.items():
            if not all(c in concepts for c in cols):
                continue
            feats = [_alive_sub(concepts[c], alive_np) for c in cols]
            r2 = _r2_ols(scores_sub, *feats)
            _append_row(
                rows,
                graph_idx=graph_idx,
                step=step,
                k=k_planted,
                target="scores",
                candidate=probe_name,
                mask="alive",
                r2=r2,
            )

        grad = grad_before if use_grad else None
        v = remove_lowest(scores, alive)
        v_int = int(v.item())

        deg_stats = removed_vertex_degree_stats(
            concepts["degree"], alive_np, v_int
        )
        slip_row = {
            "graph_idx": graph_idx,
            "step": step,
            "k": k_planted,
            "alive_count": int(alive_count),
            "initial_median_degree": initial_median_degree,
            **deg_stats,
        }
        if initial_degree_full is not None:
            init_deg_v = float(initial_degree_full[v_int])
            slip_row["initial_degree_removed"] = init_deg_v
            slip_row["is_below_initial_median_degree"] = int(
                init_deg_v < initial_median_degree
            )
            slip_row["initial_degree_percentile_weak"] = 100.0 * float(
                np.mean(initial_degree_full <= init_deg_v)
            )
        else:
            slip_row["initial_degree_removed"] = float("nan")
            slip_row["is_below_initial_median_degree"] = 0
            slip_row["initial_degree_percentile_weak"] = float("nan")
        removal_rows.append(slip_row)

        alive_after = alive.clone()
        alive_after[v_int] = False
        affected_mask = alive_after & (A[:, v_int] > 0.5)
        affected_np = affected_mask.detach().cpu().numpy().astype(bool)

        degree_before_np = deg_alive.detach().cpu().numpy()
        compat_before_np = concepts.get("compat")

        alive, deg_alive, alive_edges = train.remove_vertex_update_state(
            A, alive, deg_alive, alive_edges, v
        )
        alive_count -= 1
        if grad_cache is not None:
            grad_cache.delete_vertex_only(v)

        grad_after = grad_cache.gradient() if grad_cache is not None else None
        scores = model.update_scores(
            scores=scores_before,
            A=A,
            alive_after=alive,
            removed=v,
            grad=grad,
        )

        scores_after_np = scores.detach().cpu().numpy()
        concepts_after = _concept_dict(grad_cache, deg_alive)
        degree_after_np = deg_alive.detach().cpu().numpy()

        if int(affected_np.sum()) >= 2 and use_grad and grad_before is not None and grad_after is not None:
            if compat_before_np is None:
                compat_before_np = concepts_after.get("compat", np.zeros(n))
            update_delta = _alive_sub(scores_after_np - scores_np, affected_np)
            degree_delta = _alive_sub(degree_after_np - degree_before_np, affected_np)
            grad_delta = _alive_sub(
                grad_after.detach().cpu().numpy() - grad_before.detach().cpu().numpy(),
                affected_np,
            )
            compat_delta = _alive_sub(
                concepts_after["compat"] - compat_before_np, affected_np
            )

            delta_feats = {
                "degree_delta": degree_delta,
                "grad_delta": grad_delta,
                "compat_delta": compat_delta,
            }

            for cand in DELTA_CANDIDATES:
                sp, pr = _corr_pair(update_delta, delta_feats[cand])
                _append_row(
                    rows,
                    graph_idx=graph_idx,
                    step=step,
                    k=k_planted,
                    target="update_delta",
                    candidate=cand,
                    mask="affected_neighbors",
                    spearman=sp,
                    pearson=pr,
                )

            for probe_name, cols in DELTA_PROBES.items():
                feats = [delta_feats[c] for c in cols]
                r2 = _r2_ols(update_delta, *feats)
                _append_row(
                    rows,
                    graph_idx=graph_idx,
                    step=step,
                    k=k_planted,
                    target="update_delta",
                    candidate=probe_name,
                    mask="affected_neighbors",
                    r2=r2,
                )

    return rows, removal_rows


def _degree_write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def print_removal_degree_summary(removal_rows: list[dict]) -> None:
    if not removal_rows:
        return

    max_step_in_data = max(int(r["step"]) for r in removal_rows)
    last_step = max(NUM_STEPS - 1, max_step_in_data)

    print("\n" + "=" * 88)
    print("REMOVED VERTEX — residual degree percentile among alive (before deletion)")
    print(
        f"percentile_weak = % alive with degree <= removed; "
        f"steps 0–{last_step} (cap NUM_STEPS={NUM_STEPS}); n={NUM_GRAPHS} graphs/regime"
    )
    print("=" * 88)
    all_weak: list[float] = []
    if PRINT_FULL_STEP_TABLE:
        print(
            f"  {'step':>4s}  {'mean %ile':>10s}  {'std':>6s}  "
            f"{'P(argmin)':>9s}  {'P(>med)':>8s}  {'n':>3s}"
        )
    else:
        print(f"  (per-step table omitted; NUM_STEPS={NUM_STEPS} > 50 — see event list below)")

    for step in range(last_step + 1):
        subset = [r for r in removal_rows if int(r["step"]) == step]
        weak = [
            float(r["degree_percentile_weak"])
            for r in subset
            if r["degree_percentile_weak"] == r["degree_percentile_weak"]
        ]
        if not weak:
            if PRINT_FULL_STEP_TABLE:
                print(f"  {step:>4d}  {'n/a':>10s}  {'':>6s}  {'':>9s}  {'':>8s}  {0:>3d}")
            continue
        all_weak.extend(weak)
        if PRINT_FULL_STEP_TABLE:
            argmin_frac = float(np.mean([float(r["is_argmin_degree"]) for r in subset]))
            above_med = [int(r["is_above_median_degree"]) for r in subset]
            print(
                f"  {step:>4d}  {np.mean(weak):>10.1f}  "
                f"{np.std(weak, ddof=min(1, len(weak) - 1)):>6.1f}  "
                f"{argmin_frac:>9.2f}  {np.mean(above_med):>8.2f}  {len(weak):>3d}"
            )

    total = len(removal_rows)
    above_total = int(sum(int(r["is_above_median_degree"]) for r in removal_rows))
    print(
        f"\n  Removed degree STRICTLY > alive median: "
        f"{above_total} / {total}  ({100.0 * above_total / max(total, 1):.1f}%)"
    )
    print("  (all graphs × all recorded steps)")

    above_rows = [r for r in removal_rows if int(r["is_above_median_degree"])]
    if above_rows:
        idx_dists = [
            float(r["rank_index_from_median"])
            for r in above_rows
            if r["rank_index_from_median"] == r["rank_index_from_median"]
        ]
        deg_gaps = [
            float(r["degree_minus_median"])
            for r in above_rows
            if r["degree_minus_median"] == r["degree_minus_median"]
        ]
        print(
            f"\n  Among above-median removals only (n={len(above_rows)}):"
        )
        print(
            "    rank index = position in alive list sorted by residual degree "
            "(0=min, median index≈(n_alive-1)/2)"
        )
        if idx_dists:
            print(
                f"    mean |rank_removed - rank_median|: {np.mean(idx_dists):.1f}  "
                f"std={np.std(idx_dists, ddof=min(1, len(idx_dists) - 1)):.1f}"
            )
        if deg_gaps:
            print(
                f"    mean (removed_degree - median_degree): {np.mean(deg_gaps):.2f}  "
                f"std={np.std(deg_gaps, ddof=min(1, len(deg_gaps) - 1)):.2f}"
            )

    if all_weak:
        print(
            f"  pooled mean percentile_weak={np.mean(all_weak):.1f}  "
            f"std={np.std(all_weak, ddof=min(1, len(all_weak) - 1)):.1f}  (n={len(all_weak)} removals)"
        )


def print_weak_steps_median_check(removal_rows: list[dict]) -> None:
    """
    For steps 1–49 (weak Spearman regime): is removed residual degree below
    the alive median at that same step (both updated each iteration)?
    """
    sub = [
        r
        for r in removal_rows
        if WEAK_SPEARMAN_STEP_LO <= int(r["step"]) <= WEAK_SPEARMAN_STEP_HI
    ]
    if not sub:
        return

    n = len(sub)
    below = int(sum(int(r["is_below_median_degree"]) for r in sub))
    at_or_below = int(sum(int(r["is_at_or_below_median_degree"]) for r in sub))
    above = int(sum(int(r["is_above_median_degree"]) for r in sub))
    argmin = int(sum(int(r["is_argmin_degree"]) for r in sub))
    pct = [float(r["degree_percentile_weak"]) for r in sub]

    print("\n" + "=" * 88)
    print(
        f"STEPS {WEAK_SPEARMAN_STEP_LO}–{WEAK_SPEARMAN_STEP_HI} "
        f"(weak |Spearman| regime) vs alive median degree at that step"
    )
    print("  median & removed degree are recomputed on current alive set each step")
    print("=" * 88)
    print(f"  n = {n} removals ({NUM_GRAPHS} graphs × steps in range)")
    print(f"  removed_degree < alive_median:     {below}/{n} ({100.0 * below / n:.1f}%)")
    print(f"  removed_degree <= alive_median:    {at_or_below}/{n} ({100.0 * at_or_below / n:.1f}%)")
    print(f"  removed_degree > alive_median:     {above}/{n} ({100.0 * above / n:.1f}%)")
    print(f"  argmin(residual degree) on alive:  {argmin}/{n} ({100.0 * argmin / n:.1f}%)")
    print(
        f"  degree percentile (weak): mean={np.mean(pct):.1f}  "
        f"median={np.median(pct):.1f}"
    )
    print(
        "  → Most weak-|ρ| steps still remove below-median degree, "
        "but not min-degree (argmin ~2%)."
    )


def print_above_median_events(removal_rows: list[dict]) -> list[dict]:
    """List every (graph, step) where removed degree is strictly above alive median."""
    above_rows = sorted(
        [r for r in removal_rows if int(r["is_above_median_degree"])],
        key=lambda r: (int(r["graph_idx"]), int(r["step"])),
    )

    print("\n" + "=" * 88)
    print("ABOVE-MEDIAN REMOVALS (removed degree > alive median at that step)")
    print(f"  total events: {len(above_rows)}")
    print("=" * 88)

    if not above_rows:
        print("  (none)")
        return above_rows

    header = (
        f"  {'graph':>5s}  {'step':>5s}  {'cur%':>5s}  {'init%':>5s}  "
        f"{'cur>med':>7s}  {'init<med':>8s}  {'removed':>7s}  {'init_deg':>8s}"
    )
    print(header)
    print("  cur%=current-step degree %ile; init%=step-0 degree %ile (full graph, all alive)")
    print("  " + "-" * (len(header) - 2))
    for r in above_rows:
        init_below = int(r.get("is_below_initial_median_degree", 0))
        print(
            f"  {int(r['graph_idx']):>5d}  {int(r['step']):>5d}  "
            f"{float(r['degree_percentile_weak']):>5.1f}  "
            f"{float(r.get('initial_degree_percentile_weak', float('nan'))):>5.1f}  "
            f"{'yes':>7s}  "
            f"{'yes' if init_below else 'no':>8s}  "
            f"{float(r['removed_degree']):>7.1f}  "
            f"{float(r.get('initial_degree_removed', float('nan'))):>8.1f}"
        )

    if above_rows and "is_below_initial_median_degree" in above_rows[0]:
        n_slip = len(above_rows)
        below_init = int(sum(int(r["is_below_initial_median_degree"]) for r in above_rows))
        print(
            f"\n  Of current-step ABOVE-median slips (n={n_slip}):"
        )
        print(
            f"    step-0 (original) degree < step-0 median: "
            f"{below_init}/{n_slip} ({100.0 * below_init / n_slip:.1f}%)"
        )
        print(
            "    → if high, slip is 'low original degree' but 'high current residual degree' "
            "(degree dropped on survivors)."
        )

    by_graph: dict[int, list[int]] = defaultdict(list)
    for r in above_rows:
        by_graph[int(r["graph_idx"])].append(int(r["step"]))
    print("\n  Steps per graph (above-median only):")
    for g in sorted(by_graph):
        steps = by_graph[g]
        print(f"    graph {g}: {len(steps)} events — steps {steps}")

    return above_rows


def print_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 88)
    print("GU DEGREE + GRAD CHECK (k-less autograd, SGNN_GU + UPR, medium)")
    print("=" * 88)

    def mean_abs_sp(target: str, mask: str, candidates: tuple[str, ...]) -> None:
        print(f"\n1. mean |Spearman| — target={target}, mask={mask}")
        for cand in candidates:
            vals = []
            for r in rows:
                if (
                    r["target"] == target
                    and r["mask"] == mask
                    and r["candidate"] == cand
                    and r.get("spearman") == r.get("spearman")
                ):
                    vals.append(abs(float(r["spearman"])))
            m = float(np.mean(vals)) if vals else float("nan")
            print(f"   {cand:<32s} {m:.3f}")

    def mean_r2(target: str, mask: str, probes: dict[str, tuple[str, ...]]) -> None:
        print(f"\n2. mean R² — target={target}, mask={mask} (linear probes)")
        for name in probes:
            vals = [
                float(r["r2"])
                for r in rows
                if r["target"] == target
                and r["mask"] == mask
                and r["candidate"] == name
                and r["r2"] == r["r2"]
            ]
            m = float(np.mean(vals)) if vals else float("nan")
            print(f"   {name:<32s} {m:.3f}")

    mean_abs_sp("scores", "alive", SCORE_CANDIDATES)
    mean_r2("scores", "alive", SCORE_PROBES)

    print("\n--- update_delta on affected neighbors (GU neighbor gate) ---")
    mean_abs_sp("update_delta", "affected_neighbors", DELTA_CANDIDATES)
    mean_r2("update_delta", "affected_neighbors", DELTA_PROBES)

    print("\n" + "-" * 88)
    print("Notes:")
    print("  • degree_delta is -1 on every affected neighbor → correlation undefined (nan).")
    print("  • degree/grad/Ap/Bp/compat often share one ranking → similar |Spearman|.")
    print("  • degree+grad+Ap+Bp+compat R² nan when probe features are collinear.")
    print("Readout (your run):")
    print("  • scores: weak ~0.27 |ρ|; R² ~0.10–0.12, little gain from +grad → not degree+grad")
    print("  • update_delta: compat_delta |ρ| ~0.24 best; R² ~0.02 → weak local GU signal")
    print()


def _filter_removal_band(
    removal_rows: list[dict],
    step_lo: int | None,
    step_hi: int | None,
) -> list[dict]:
    if step_lo is None or step_hi is None:
        return removal_rows
    return [r for r in removal_rows if step_lo <= int(r["step"]) <= step_hi]


def _slip_counts(removal_rows: list[dict], field: str) -> tuple[int, int]:
    n = len(removal_rows)
    above = int(sum(int(r[field]) for r in removal_rows))
    return above, n


def _percentile_slip_field(threshold: float = DEGREE_PERCENTILE_SLIP_THRESHOLD) -> str:
    if threshold == DEGREE_PERCENTILE_SLIP_THRESHOLD:
        return "is_above_p15_degree"
    raise ValueError(f"No slip field for percentile threshold {threshold}")


def median_removal_stats(removal_rows: list[dict]) -> dict[str, float | int]:
    total = len(removal_rows)
    if total == 0:
        return {"total": 0}
    weak_lo, weak_hi = WEAK_SPEARMAN_STEP_LO, WEAK_SPEARMAN_STEP_HI
    weak = [r for r in removal_rows if weak_lo <= int(r["step"]) <= weak_hi]
    above_all = int(sum(int(r["is_above_median_degree"]) for r in removal_rows))
    above_weak = int(sum(int(r["is_above_median_degree"]) for r in weak))
    argmin_weak = int(sum(int(r["is_argmin_degree"]) for r in weak))
    pct_weak = [float(r["degree_percentile_weak"]) for r in weak]
    slip_steps = sorted(
        {int(r["step"]) for r in removal_rows if int(r["is_above_median_degree"])}
    )
    return {
        "total": total,
        "above_all": above_all,
        "above_all_pct": 100.0 * above_all / total,
        "weak_n": len(weak),
        "above_weak": above_weak,
        "above_weak_pct": 100.0 * above_weak / max(len(weak), 1),
        "argmin_weak_pct": 100.0 * argmin_weak / max(len(weak), 1),
        "pct_weak_mean": float(np.mean(pct_weak)) if pct_weak else float("nan"),
        "slip_step_lo": min(slip_steps) if slip_steps else -1,
        "slip_step_hi": max(slip_steps) if slip_steps else -1,
        "slip_step_count": len(slip_steps),
    }


def build_above_median_by_graph_rows(
    spec: TrainSpec,
    removal_rows: list[dict],
    graphs: list[dict],
) -> list[dict]:
    """Per-graph above-median removal rate for each step band."""
    by_graph: dict[int, list[dict]] = defaultdict(list)
    for row in removal_rows:
        by_graph[int(row["graph_idx"])].append(row)

    meta = {int(g["graph_idx"]): g for g in graphs}
    out: list[dict] = []
    for graph_idx in sorted(by_graph):
        gmeta = meta[graph_idx]
        graph_rows = by_graph[graph_idx]
        for band, step_lo, step_hi in STEP_BANDS:
            sub = _filter_removal_band(graph_rows, step_lo, step_hi)
            above, n = _slip_counts(sub, "is_above_median_degree")
            rate = (100.0 * above / n) if n else float("nan")
            out.append(
                {
                    "model_name": spec.name,
                    "architecture": spec.architecture,
                    "regime": REGIME,
                    "step_band": band,
                    "graph_idx": graph_idx,
                    "seed": gmeta["seed"],
                    "k_planted": gmeta["k_planted"],
                    "num_removals": n,
                    "num_above_median": above,
                    "rate_above_median_pct": rate,
                }
            )
    return out


def build_above_median_aggregate_rows(
    by_graph_rows: list[dict],
) -> list[dict]:
    """Mean/std/min/max of per-graph rates + pooled rate, grouped by model and band."""
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in by_graph_rows:
        key = (
            row["model_name"],
            row["architecture"],
            row["regime"],
            row["step_band"],
        )
        grouped[key].append(row)

    out: list[dict] = []
    for (model_name, architecture, regime, band), rows in sorted(grouped.items()):
        rates = np.asarray(
            [float(r["rate_above_median_pct"]) for r in rows],
            dtype=np.float64,
        )
        n_removals = int(sum(int(r["num_removals"]) for r in rows))
        n_above = int(sum(int(r["num_above_median"]) for r in rows))
        pooled_rate = 100.0 * n_above / max(n_removals, 1)
        out.append(
            {
                "model_name": model_name,
                "architecture": architecture,
                "regime": regime,
                "step_band": band,
                "n_graphs": len(rows),
                "num_removals": n_removals,
                "num_above_median": n_above,
                "rate_pooled_pct": pooled_rate,
                "rate_per_graph_mean_pct": float(np.mean(rates)),
                "rate_per_graph_std_pct": float(np.std(rates, ddof=min(1, len(rates) - 1))),
                "rate_per_graph_min_pct": float(np.min(rates)),
                "rate_per_graph_max_pct": float(np.max(rates)),
            }
        )
    return out


def print_above_median_metric_report(
    by_graph_rows: list[dict],
    aggregate_rows: list[dict],
) -> None:
    print("\n" + "=" * 88)
    print("ABOVE-MEDIAN REMOVAL RATE (removed current degree > current alive median)")
    print(
        f"  regime={REGIME}  graphs={NUM_GRAPHS}  steps=0–{NUM_STEPS - 1}  "
        "rate = % removals strictly above median at that step"
    )
    print("=" * 88)

    for band in [b[0] for b in STEP_BANDS]:
        band_agg = [r for r in aggregate_rows if r["step_band"] == band]
        if not band_agg:
            continue
        print(f"\n  step band: {band}")
        print(
            f"    {'architecture':<10s}  {'pooled %':>9s}  "
            f"{'mean±std (per graph)':>22s}  {'min':>7s}  {'max':>7s}  "
            f"{'count':>12s}"
        )
        for row in band_agg:
            mean = float(row["rate_per_graph_mean_pct"])
            std = float(row["rate_per_graph_std_pct"])
            print(
                f"    {row['architecture']:<10s}  "
                f"{float(row['rate_pooled_pct']):>8.2f}%  "
                f"{mean:>8.2f}% ± {std:<6.2f}%  "
                f"{float(row['rate_per_graph_min_pct']):>6.2f}%  "
                f"{float(row['rate_per_graph_max_pct']):>6.2f}%  "
                f"{int(row['num_above_median']):>5d}/{int(row['num_removals']):<5d}"
            )

        print(f"    per-graph breakdown ({band}):")
        print(
            f"      {'graph':>5s}  {'arch':>6s}  {'k':>3s}  "
            f"{'above':>5s}  {'n':>5s}  {'rate %':>8s}"
        )
        band_graph = sorted(
            [r for r in by_graph_rows if r["step_band"] == band],
            key=lambda r: (r["architecture"], int(r["graph_idx"])),
        )
        for row in band_graph:
            arch = "GU" if "GU" in row["architecture"] else "U"
            print(
                f"      {int(row['graph_idx']):>5d}  {arch:>6s}  "
                f"{int(row['k_planted']):>3d}  "
                f"{int(row['num_above_median']):>5d}  {int(row['num_removals']):>5d}  "
                f"{float(row['rate_above_median_pct']):>8.2f}"
            )


def write_above_median_metrics(
    by_graph_rows: list[dict],
    aggregate_rows: list[dict],
) -> None:
    _degree_write_csv(ABOVE_MEDIAN_BY_GRAPH_CSV, by_graph_rows)
    _degree_write_csv(ABOVE_MEDIAN_AGGREGATE_CSV, aggregate_rows)
    print(f"\nWrote {len(by_graph_rows)} rows → {ABOVE_MEDIAN_BY_GRAPH_CSV}")
    print(f"Wrote {len(aggregate_rows)} rows → {ABOVE_MEDIAN_AGGREGATE_CSV}")


def build_above_p15_by_graph_rows(
    spec: TrainSpec,
    removal_rows: list[dict],
    graphs: list[dict],
) -> list[dict]:
    """Per-graph rate of removals with degree percentile (weak) strictly above 15."""
    by_graph: dict[int, list[dict]] = defaultdict(list)
    for row in removal_rows:
        by_graph[int(row["graph_idx"])].append(row)

    meta = {int(g["graph_idx"]): g for g in graphs}
    out: list[dict] = []
    thr = DEGREE_PERCENTILE_SLIP_THRESHOLD
    for graph_idx in sorted(by_graph):
        gmeta = meta[graph_idx]
        graph_rows = by_graph[graph_idx]
        for band, step_lo, step_hi in STEP_BANDS:
            sub = _filter_removal_band(graph_rows, step_lo, step_hi)
            above, n = _slip_counts(sub, "is_above_p15_degree")
            rate = (100.0 * above / n) if n else float("nan")
            out.append(
                {
                    "model_name": spec.name,
                    "architecture": spec.architecture,
                    "regime": REGIME,
                    "step_band": band,
                    "percentile_threshold": thr,
                    "graph_idx": graph_idx,
                    "seed": gmeta["seed"],
                    "k_planted": gmeta["k_planted"],
                    "num_removals": n,
                    "num_above_p15": above,
                    "rate_above_p15_pct": rate,
                }
            )
    return out


def build_above_p15_aggregate_rows(by_graph_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in by_graph_rows:
        key = (
            row["model_name"],
            row["architecture"],
            row["regime"],
            row["step_band"],
        )
        grouped[key].append(row)

    out: list[dict] = []
    for (model_name, architecture, regime, band), rows in sorted(grouped.items()):
        rates = np.asarray(
            [float(r["rate_above_p15_pct"]) for r in rows],
            dtype=np.float64,
        )
        n_removals = int(sum(int(r["num_removals"]) for r in rows))
        n_above = int(sum(int(r["num_above_p15"]) for r in rows))
        pooled_rate = 100.0 * n_above / max(n_removals, 1)
        out.append(
            {
                "model_name": model_name,
                "architecture": architecture,
                "regime": regime,
                "step_band": band,
                "percentile_threshold": DEGREE_PERCENTILE_SLIP_THRESHOLD,
                "n_graphs": len(rows),
                "num_removals": n_removals,
                "num_above_p15": n_above,
                "rate_pooled_pct": pooled_rate,
                "rate_per_graph_mean_pct": float(np.mean(rates)),
                "rate_per_graph_std_pct": float(np.std(rates, ddof=min(1, len(rates) - 1))),
                "rate_per_graph_min_pct": float(np.min(rates)),
                "rate_per_graph_max_pct": float(np.max(rates)),
            }
        )
    return out


def print_above_p15_metric_report(
    by_graph_rows: list[dict],
    aggregate_rows: list[dict],
) -> None:
    thr = DEGREE_PERCENTILE_SLIP_THRESHOLD
    print("\n" + "=" * 88)
    print(
        f"ABOVE-{thr}th PERCENTILE REMOVAL RATE "
        f"(degree_percentile_weak > {thr} among alive)"
    )
    print(
        f"  regime={REGIME}  graphs={NUM_GRAPHS}  steps=0–{NUM_STEPS - 1}  "
        f"GU target: 0% slips (always at or below {thr}th percentile)"
    )
    print("=" * 88)

    for band in [b[0] for b in STEP_BANDS]:
        band_agg = [r for r in aggregate_rows if r["step_band"] == band]
        if not band_agg:
            continue
        print(f"\n  step band: {band}")
        print(
            f"    {'architecture':<10s}  {'pooled %':>9s}  "
            f"{'mean±std (per graph)':>22s}  {'min':>7s}  {'max':>7s}  "
            f"{'count':>12s}"
        )
        for row in band_agg:
            mean = float(row["rate_per_graph_mean_pct"])
            std = float(row["rate_per_graph_std_pct"])
            print(
                f"    {row['architecture']:<10s}  "
                f"{float(row['rate_pooled_pct']):>8.2f}%  "
                f"{mean:>8.2f}% ± {std:<6.2f}%  "
                f"{float(row['rate_per_graph_min_pct']):>6.2f}%  "
                f"{float(row['rate_per_graph_max_pct']):>6.2f}%  "
                f"{int(row['num_above_p15']):>5d}/{int(row['num_removals']):<5d}"
            )


def _per_graph_pct_removals_le_t(removal_rows: list[dict], t: float) -> np.ndarray:
    """
    For each graph_idx, fraction of removals with degree_percentile_weak <= t, as 0–100%.
    """
    by_g: dict[int, list[float]] = defaultdict(list)
    for r in removal_rows:
        by_g[int(r["graph_idx"])].append(float(r["degree_percentile_weak"]))
    rates: list[float] = []
    for gi in sorted(by_g.keys()):
        arr = np.asarray(by_g[gi], dtype=np.float64)
        rates.append(100.0 * float(np.mean(arr <= t)))
    return np.asarray(rates, dtype=np.float64)


def print_percentile_le_t_mean_std_table(removal_by_model: dict[str, list[dict]]) -> None:
    """
    Simple table: threshold T vs mean ± std (across graphs) of % removals with
    degree_percentile_weak <= T, for SGNN+GU and SGNN+U on the same instances.
    """
    gu_rows = removal_by_model[GU_SPEC.name]
    u_rows = removal_by_model[U_SPEC.name]

    print("\n" + "=" * 88)
    print(
        "REMOVALS WITH degree_percentile_weak <= T "
        f"(mean ± std % across {NUM_GRAPHS} graphs; {NUM_STEPS} steps/graph)"
    )
    print("=" * 88)
    header = (
        f"  {'≤T (%)':>8s}  "
        f"{'SGNN+GU mean ± std':>24s}  "
        f"{'SGNN+U mean ± std':>24s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    csv_rows: list[dict[str, float | int]] = []
    for t in PERCENTILE_TABLE_THRESHOLDS:
        g_gu = _per_graph_pct_removals_le_t(gu_rows, float(t))
        g_u = _per_graph_pct_removals_le_t(u_rows, float(t))
        m_gu, s_gu = float(np.mean(g_gu)), float(np.std(g_gu, ddof=min(1, len(g_gu) - 1)))
        m_u, s_u = float(np.mean(g_u)), float(np.std(g_u, ddof=min(1, len(g_u) - 1)))
        print(
            f"  {t:8d}  "
            f"{m_gu:>10.2f}% ± {s_gu:<8.2f}%  "
            f"{m_u:>10.2f}% ± {s_u:<8.2f}%"
        )
        csv_rows.append(
            {
                "threshold_T": int(t),
                "gu_mean_pct": m_gu,
                "gu_std_pct": s_gu,
                "u_mean_pct": m_u,
                "u_std_pct": s_u,
            }
        )

    _degree_write_csv(PERCENTILE_LE_T_TABLE_CSV, csv_rows)
    print(f"\nWrote {len(csv_rows)} rows → {PERCENTILE_LE_T_TABLE_CSV}")


def print_gu_percentile_coverage(removal_rows: list[dict]) -> None:
    """For GU only: fraction of removals at or below each percentile threshold."""
    pct = np.asarray(
        [float(r["degree_percentile_weak"]) for r in removal_rows],
        dtype=np.float64,
    )
    print("\n" + "=" * 88)
    print("GU — degree_percentile_weak coverage (pooled over all removals)")
    print("=" * 88)
    print(
        f"  n={len(pct)}  mean={float(np.mean(pct)):.2f}%  "
        f"median={float(np.median(pct)):.2f}%  max={float(np.max(pct)):.2f}%"
    )
    print(f"  {'T':>4}  {'% removals <= T':>16}")
    for t in PERCENTILE_THRESHOLDS:
        print(f"  {t:4d}  {100.0 * float(np.mean(pct <= t)):15.2f}%")


def write_above_p15_metrics(
    by_graph_rows: list[dict],
    aggregate_rows: list[dict],
) -> None:
    _degree_write_csv(ABOVE_P15_BY_GRAPH_CSV, by_graph_rows)
    _degree_write_csv(ABOVE_P15_AGGREGATE_CSV, aggregate_rows)
    print(f"\nWrote {len(by_graph_rows)} rows → {ABOVE_P15_BY_GRAPH_CSV}")
    print(f"Wrote {len(aggregate_rows)} rows → {ABOVE_P15_AGGREGATE_CSV}")


def print_median_removal_compare(by_model: dict[str, list[dict]]) -> None:
    print("\n" + "=" * 88)
    print("MEDIAN REMOVAL COMPARISON (same graphs, k-less autograd UPR)")
    print("=" * 88)
    print(
        f"  {'model':<40s}  {'>med all':>10s}  "
        f"{f'>med {WEAK_SPEARMAN_STEP_LO}-{WEAK_SPEARMAN_STEP_HI}':>14s}  "
        f"{'argmin weak':>11s}  {'mean %ile weak':>14s}  {'slip steps':>12s}"
    )
    for name, rows in by_model.items():
        s = median_removal_stats(rows)
        slip_range = (
            "none"
            if s.get("above_all", 0) == 0
            else f"{s['slip_step_lo']}–{s['slip_step_hi']} ({s['slip_step_count']})"
        )
        print(
            f"  {name:<40s}  "
            f"{s['above_all']:>4d}/{s['total']:<4d}  "
            f"{s['above_weak']:>4d}/{s['weak_n']:<4d}  "
            f"{s['argmin_weak_pct']:>10.1f}%  "
            f"{s['pct_weak_mean']:>14.2f}  "
            f"{slip_range:>12s}"
        )


def run_spec(
    spec: TrainSpec,
    device: torch.device,
    graphs: list[dict],
) -> tuple[list[dict], list[dict]]:
    model = train.load_model_checkpoint(spec, device)
    model.eval()

    print("\n" + "#" * 88)
    print(f"MODEL: {spec.name} ({spec.architecture}, training={spec.training_way})")
    print("#" * 88)

    all_rows: list[dict] = []
    all_removal_rows: list[dict] = []
    for graph in graphs:
        trace, removal = collect_degree_grad_check(
            model,
            graph["A"],
            graph["k_planted"],
            graph_idx=graph["graph_idx"],
            max_steps=NUM_STEPS,
        )
        for row in trace:
            row["model_name"] = spec.name
        for row in removal:
            row["model_name"] = spec.name
        all_rows.extend(trace)
        all_removal_rows.extend(removal)
        print(
            f"  graph {graph['graph_idx']}: k_planted={graph['k_planted']} "
            f"seed={graph['seed']} corr_rows={len(trace)} removal_rows={len(removal)}"
        )

    out_csv, out_rem, out_above = output_paths_for(spec)
    _degree_write_csv(out_csv, all_rows)
    _degree_write_csv(out_rem, all_removal_rows)
    print(f"\nWrote {len(all_rows)} rows → {out_csv}")
    print(f"Wrote {len(all_removal_rows)} rows → {out_rem}")
    print_removal_degree_summary(all_removal_rows)
    print_weak_steps_median_check(all_removal_rows)
    above_events = print_above_median_events(all_removal_rows)
    _degree_write_csv(out_above, above_events)
    print(f"Wrote {len(above_events)} above-median events → {out_above}")
    if spec.use_gradient_in_updater:
        print_summary(all_rows)
    return all_rows, all_removal_rows


_DEGREE_SEED = 0


def _bind_degree_paths() -> None:
    global ABOVE_MEDIAN_BY_GRAPH_CSV, ABOVE_MEDIAN_AGGREGATE_CSV
    global ABOVE_P15_BY_GRAPH_CSV, ABOVE_P15_AGGREGATE_CSV, PERCENTILE_LE_T_TABLE_CSV
    global _DEGREE_SEED
    root = train.OUTPUT_DIR
    _DEGREE_SEED = train.SEED
    ABOVE_MEDIAN_BY_GRAPH_CSV = root / "above_median_rate_by_graph.csv"
    ABOVE_MEDIAN_AGGREGATE_CSV = root / "above_median_rate_aggregate.csv"
    ABOVE_P15_BY_GRAPH_CSV = root / "above_p15_rate_by_graph.csv"
    ABOVE_P15_AGGREGATE_CSV = root / "above_p15_rate_aggregate.csv"
    PERCENTILE_LE_T_TABLE_CSV = root / "percentile_le_t_mean_std_table.csv"


def cmd_degree(args: argparse.Namespace) -> None:
    train.apply_run_config(train_regime=args.train_regime)
    _bind_degree_paths()

    device = torch.device(train.DEVICE)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("GU / U degree+grad check (Tables 4–5)")
    print(f"output_dir={train.OUTPUT_DIR}")
    print(
        f"regime={REGIME} planted_k∈[{K_MIN},{K_MAX}] (eval only) "
        f"n={train.TEST_N} graphs={NUM_GRAPHS} steps={NUM_STEPS}"
    )
    print(f"models: {', '.join(s.name for s in RUN_SPECS)}")

    graphs: list[dict] = []
    for graph_idx in range(NUM_GRAPHS):
        seed = instance_seed(graph_idx)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        k_planted = random.randint(K_MIN, K_MAX)
        A, _planted = train.make_planted_clique(train.TEST_N, k_planted, device)
        graphs.append(
            {
                "A": A,
                "k_planted": k_planted,
                "graph_idx": graph_idx,
                "seed": seed,
            }
        )

    removal_by_model: dict[str, list[dict]] = {}
    for spec in RUN_SPECS:
        _rows, removal = run_spec(spec, device, graphs)
        removal_by_model[spec.name] = removal

    print_median_removal_compare(removal_by_model)

    by_graph_rows: list[dict] = []
    for spec in RUN_SPECS:
        by_graph_rows.extend(
            build_above_median_by_graph_rows(spec, removal_by_model[spec.name], graphs)
        )
    aggregate_rows = build_above_median_aggregate_rows(by_graph_rows)
    write_above_median_metrics(by_graph_rows, aggregate_rows)
    print_above_median_metric_report(by_graph_rows, aggregate_rows)

    p15_by_graph: list[dict] = []
    for spec in RUN_SPECS:
        p15_by_graph.extend(
            build_above_p15_by_graph_rows(spec, removal_by_model[spec.name], graphs)
        )
    p15_aggregate = build_above_p15_aggregate_rows(p15_by_graph)
    write_above_p15_metrics(p15_by_graph, p15_aggregate)
    print_above_p15_metric_report(p15_by_graph, p15_aggregate)
    print_percentile_le_t_mean_std_table(removal_by_model)
    print_gu_percentile_coverage(removal_by_model[GU_SPEC.name])



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_regime_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--train-regime",
        choices=list(train.REGIMES),
        default="medium",
        help="Checkpoint directory runs/sgnn_paper_{regime}/",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    snap = sub.add_parser(
        "snap",
        help="SNAP zero-shot eval (Table 6)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_regime_arg(snap)
    snap.add_argument(
        "--layers",
        type=int,
        default=None,
        help="Override GNN layers (e.g. 8 for L8 checkpoints). Default: 4.",
    )
    snap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs suffix (e.g. 1200 for _E1200 checkpoints). Default: none.",
    )
    snap.add_argument(
        "--datasets",
        nargs="+",
        default=list(SNAP_DATASETS),
        help="SNAP stems under datasets/*.jsonl",
    )
    snap.add_argument(
        "--decoders",
        nargs="+",
        default=("upr",),
        help="Decoder names (ldr needs no checkpoint, others need model)",
    )
    snap.add_argument(
        "--models",
        nargs="+",
        choices=("sgnn", "u", "gu", "dgu", "both"),
        default=("both",),
        help="Checkpoints to evaluate: sgnn, u, gu, dgu, or both (all four)",
    )
    snap.set_defaults(func=cmd_snap)

    bench = sub.add_parser(
        "bench",
        help="Decoding wall-clock on random ER graphs (Table 3 / App. K)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_regime_arg(bench)
    bench.add_argument(
        "--ns",
        type=int,
        nargs="+",
        default=None,
        help=f"Graph sizes (default: {' '.join(map(str, BENCH_NS))})",
    )
    bench.add_argument(
        "--instances",
        type=int,
        default=10,
        help="Random graphs per (n, architecture, decoder)",
    )
    bench.set_defaults(func=cmd_bench)

    degree = sub.add_parser(
        "degree",
        help="UPR removal vs degree/gradient (Tables 4–5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_regime_arg(degree)
    degree.set_defaults(func=cmd_degree)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
