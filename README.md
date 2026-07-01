# n_squared_clique

Reproduction code for *The Missing Signal: Gradient Injection Recovers O(n²) Residual Decoding in GNNs* (`main.pdf`).

All commands run from the **repository root**. Python code lives under `src/`. Results go to `runs/sgnn_paper_{easy|medium|hard}/`.

---

## Contents

1. [Setup](#setup)
2. [Repository layout](#repository-layout)
3. [Paper protocol](#paper-protocol)
4. [Quick start](#quick-start)
5. [Paper → commands](#paper--commands)
6. [Train (`src.train`)](#train-srctrain)
7. [Experiments (`src.experiments`)](#experiments-srcexperiments)
8. [Plots (`src.gu_plots`)](#plots-srcgu_plots)
9. [Shared concepts](#shared-concepts)
10. [Output paths](#output-paths)
11. [Notes](#notes)

---

## Setup

### Requirements

- **Python 3.10+** (3.11 tested)
- **CUDA GPU** recommended for training (CPU works but is slow)
- ~2 GB disk for venv; more for `runs/` and datasets

Pinned packages in `requirements.txt`:

| Package | Role |
|---------|------|
| `torch` | GNN training and eval |
| `numpy`, `scipy` | Graphs and statistics |
| `scikit-learn` | PCA in plots |
| `matplotlib`, `plotly` | Paper figures |
| `tqdm` | Progress bars |

### Install (recommended)

```bash
cd /path/to/n_sqaured_clique
./setup_env.sh
source .venv/bin/activate
```

`setup_env.sh` creates `.venv`, upgrades `pip`, and runs `pip install -r requirements.txt`.

### Manual install

```bash
cd /path/to/n_sqaured_clique
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Datasets (SNAP, Table 6)

SNAP graphs are **not** bundled in this repo. Point `datasets/` at your copy:

```bash
# Example: share CliqueEvaluation datasets
ln -sf ../CliqueEvaluation/datasets datasets
```

You need:

- `datasets/<stem>.jsonl` for each SNAP stem (see [SNAP eval](#snap--table-6))
- `runs/sgnn_paper_medium/snap_gt_cache/<stem>.jsonl` — precomputed max-clique labels (no PMC at eval time)

### Verify install

```bash
source .venv/bin/activate
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -m src.train --help
python -m src.experiments --help
python -m src.gu_plots --help
```

---

## Repository layout

```
src/
  train.py            # Train + planted-clique eval (Tables 2–3)
  models.py           # ResidualGNN (+ forward); SGNN / SGNN+U / SGNN+GU in docstring
  experiments.py      # snap | bench | degree (Tables 3–6, mechanism)
  gu_plots.py         # Figures 2–3 (HTML + PDF)
runs/sgnn_paper_{easy|medium|hard}/   # outputs per --train-regime
datasets/                             # SNAP jsonl (symlink)
requirements.txt
setup_env.sh
main.pdf                              # paper (if present)
```

---

## Paper protocol

Settings aligned with **Sec. 3.4** in `main.pdf` (see `train.py` constants):

| Setting | Value |
|---------|--------|
| Graph size | n = 1000, G(n, ½) + planted clique |
| Regimes | Easy k∈[62,100], Medium k∈[36,61], Hard k∈[20,35] |
| Epochs | 64 |
| Training graphs / epoch | 1000 (one AdamW step per graph) |
| Held-out eval / regime | 1000 instances |
| Backbone | L=4, hidden=64, tanh, agg. **sum_norm_n** (1/(n−1)) |
| SIT | `self_predictor_lpr`, 300 teacher steps |
| Main checkpoints | **Medium-trained** → `runs/sgnn_paper_medium/` |

**K-less rule:** true planted size *k* is used only for **generating** graphs and **reporting** metrics — never in the model, loss, GU cache, or decoders.

---

## Quick start

Medium-trained run (paper default):

```bash
source .venv/bin/activate

# 1) Train five checkpoints + planted eval (long)
python -m src.train --train-regime medium

# 2) Or eval only if checkpoints already exist
python -m src.train --train-regime medium --skip-training

# 3) Other paper tables / figures
python -m src.experiments snap --train-regime medium
python -m src.experiments bench --train-regime medium
python -m src.experiments degree --train-regime medium
python -m src.gu_plots --train-regime medium
```

---

## Paper → commands

| Paper artifact | Command |
|----------------|---------|
| **Table 2–3** train + planted eval | `python -m src.train --train-regime medium` |
| Easy / hard training | `--train-regime easy` or `hard` |
| Eval only | add `--skip-training` |
| **Table 6** SNAP U-LPR | `python -m src.experiments snap --train-regime medium` |
| **Table 3 / App. K** timing | `python -m src.experiments bench --train-regime medium --instances 1000` |
| **Tables 4–5** mechanism | `python -m src.experiments degree --train-regime medium` |
| **Fig. 2–3** HTML + PDF | `python -m src.gu_plots --train-regime medium --format both` |
| **Fig. 2–3** HTML only | `python -m src.gu_plots --format html` |
| **Fig. 2–3** PDF only | `python -m src.gu_plots --format paper --extra-pca-all-arch` |
| Train PCA grids (diagnostic) | `python -m src.train --pca-plots` |

---

## Train (`src.train`)

Trains **SGNN**, **SGNN+U**, **SGNN+GU** (objective + SIT where applicable), then evaluates decoders on synthetic planted graphs.

```bash
python -m src.train --help
```

| Flag | Default | Effect |
|------|---------|--------|
| `--train-regime` | `medium` | Training *k* range + `runs/sgnn_paper_{regime}/` |
| `--skip-training` | off | Load `.pt` checkpoints and eval only |
| `--run-degree-corr` | off | Quick inline UPR–degree CSV (prefer `experiments degree`) |
| `--pca-plots` | off | All-architecture PCA grids under `plots/pca/` |

**Checkpoints** (`models/`):

| File | Architecture | Training |
|------|--------------|----------|
| `SGNN__objective_train.pt` | SGNN | objective |
| `SGNN_U__objective_train.pt` | SGNN+U | objective |
| `SGNN_U__self_predictor_lpr_train.pt` | SGNN+U | SIT |
| `SGNN_GU__objective_train.pt` | SGNN+GU | objective |
| `SGNN_GU__self_predictor_lpr_train.pt` | SGNN+GU | SIT |

**Main outputs:** `per_instance_results.csv`, `aggregate_results.csv` (Table 2), `run_config.json`.

---

## Experiments (`src.experiments`)

Post-training tools. **Requires checkpoints** from `src.train`.

```bash
python -m src.experiments --help
python -m src.experiments snap --help
python -m src.experiments bench --help
python -m src.experiments degree --help
```

All subcommands accept `--train-regime {easy,medium,hard}` (default `medium`).

### `snap` — Table 6

Zero-shot **SGNN+U** / **SGNN+GU** (SIT) on six SNAP graphs with cached max-clique labels.

```bash
python -m src.experiments snap --train-regime medium
python -m src.experiments snap --decoders upr --models both
python -m src.experiments snap --datasets com-orkut --models gu
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--datasets` | twitter, collab, imdb_binary, com-youtube, com-orkut, facebook | Stems under `datasets/*.jsonl` |
| `--decoders` | `upr` | `one_pass`, `rerun_pruned`, `lpr`, `upr` |
| `--models` | `both` | `u`, `gu`, or `both` |

**Writes:** `snap_per_instance_results.csv`, `snap_aggregate_results.csv`.

### `bench` — Table 3 / Appendix K

Wall-clock decoding on **unplanted** random G(n, ½). Prints a table to **stdout** (no CSV).

```bash
python -m src.experiments bench --train-regime medium
python -m src.experiments bench --ns 1000 --instances 100
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--ns` | 500 1000 1500 2000 5000 | Graph orders |
| `--instances` | 10 | Graphs per (n, model, decoder) |

Stdout labels: `topk`→`one_pass`, `rerun`→`rerun_pruned`, `ulpr`→`upr`.

### `degree` — Tables 4–5

UPR removal statistics (degree vs gradient) on 1000 medium-*k* graphs × 500 steps. Heavy run.

```bash
python -m src.experiments degree --train-regime medium
```

**Writes:** `gu_degree_grad_check__*.csv`, `above_median_rate_*.csv`, `percentile_le_t_mean_std_table.csv`, etc.

Eval *k* band is set in `experiments.py` (`REGIME=medium`, k∈[36,61]) unless you edit that section.

---

## Plots (`src.gu_plots`)

Figures 2–3 and train-time PCA grid helpers.

```bash
python -m src.gu_plots --help
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--train-regime` | `medium` | Checkpoint directory |
| `--format` | `both` | `html` (Plotly), `paper` (PDF), or `both` |
| `--graphs` | format-specific | Graph indices (default: html→`[1]`, paper→`[0,1]`) |
| `--extra-pca-all-arch` | off | PDF: extra PCA for all five checkpoints |

**Paper figures (SGNN+U / SGNN+GU SIT):**

1. PCA of GNN **hidden** states (PC1 vs PC2, colored by residual degree).
2. UPR panels at steps 0, 1, 5, 10, 300 (score vs degree).

```bash
python -m src.gu_plots --train-regime medium
python -m src.gu_plots --format html --graphs 1
python -m src.gu_plots --format paper --extra-pca-all-arch
```

**Outputs:**

- HTML: `runs/.../plots/*.html`
- PDF: `runs/.../plots/paper/*.pdf`
- Train grids (`train --pca-plots`): `runs/.../plots/pca/`

---

## Shared concepts

### `--train-regime`

| Regime | Training *k* | Directory |
|--------|--------------|-----------|
| `easy` | 62–100 | `runs/sgnn_paper_easy/` |
| `medium` | 36–61 | `runs/sgnn_paper_medium/` |
| `hard` | 20–35 | `runs/sgnn_paper_hard/` |

### Decoder names (code vs paper)

| Code | Paper / bench alias | Typical cost |
|------|---------------------|--------------|
| `one_pass` | topk | O(n²) |
| `rerun_pruned` | rerun | O(n³) |
| `lpr` | LPR | O(n²) |
| `upr` | U-LPR | O(n²) |

Planted eval in `train` uses fixed decoders per architecture (see `DECODERS_BY_ARCH` in `train.py`).

---

## Output paths

Under `runs/sgnn_paper_medium/` (example):

| Path | Paper |
|------|-------|
| `models/*.pt` | Checkpoints |
| `aggregate_results.csv` | Table 2 |
| `snap_aggregate_results.csv` | Table 6 |
| `plots/*.html` | Fig. 2–3 (interactive) |
| `plots/paper/*.pdf` | Fig. 2–3 (print) |
| `snap_gt_cache/*.jsonl` | SNAP ground truth |

---

## Notes

- Checkpoints in `runs/sgnn_paper_medium/` may be from an **older** protocol (e.g. 300 epochs). For exact Sec. 3.4 numbers, retrain with current defaults (`EPOCHS=64`, `TRAIN_GRAPHS_PER_EPOCH=1000`).
- Table 6 column **”SGNN Rerun †”** in the paper is from prior work [11], not this repo.
- `gu_plots` / `experiments degree` use **medium** eval *k* in filenames even if you load easy/hard checkpoints — edit `REGIME`, `K_MIN`, `K_MAX` in `gu_plots.py` or the degree section of `experiments.py`.
- **Training is lightweight and easy to reproduce** — a full medium-regime run takes a few hours on a single GPU. If you prefer to skip training entirely, pre-trained weights are available on request: open a GitHub issue and we will share them.
