# KAN ESS — KAN-based Battery Voltage Model for HEMS

Thesis project: *Synthesis of a KAN ESS model for an online REC HEMS*
CIPAR Labs, Sapienza University of Rome

A [Kolmogorov-Arnold Network (KAN)](https://arxiv.org/abs/2404.19756) trained to model
terminal voltage of an ePLB C020 Li-ion cell.  Task: **[I, SoC, T, ΔI] → V**.

After training, the learned B-spline coefficients are exported to `splines_kan.json`
and evaluated at runtime using pure NumPy — no PyTorch dependency at inference time.

---

## Repository structure

```
KAN_ESS/
├── dataset/              Preprocessed RW9–RW12.mat  (~34 MB total, committed)
├── models/               All training runs (committed)
│   └── KAN_{timestamp}/  One directory per run — weights, splines, plots, log
├── training/
│   ├── preprocess.py     Raw .mat parser + SoC Coulomb counting
│   ├── data_loader.py    Feature construction [I, SoC, T, dI] -> V, normalisation
│   ├── kan_model.py      KAN architecture, B-spline export, NumPy inference
│   └── train.py          Training script (entry point)
├── ESS_kan.py            Drop-in HEMS ESS subclass (NumPy inference only)
├── build_dataset.py      One-time preprocessing of raw RW*.mat files
├── requirements.txt
└── README.md
```

---

## Quickstart (new workstation)

### Prerequisites

- **Python 3.10 or later** (tested on Python 3.12)
- **Git** — required both to clone the repo and to install `efficient-kan` from GitHub

Verify before starting:
```bash
python --version   # must be 3.10+
git --version
```

### 1. Clone the repository

```bash
git clone https://github.com/ahmet-kemal-ates/KAN_ESS.git
cd KAN_ESS
```

### 2. Create a virtual environment

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate

# Windows (Command Prompt)
python -m venv .venv
.venv\Scripts\activate

# Windows (PowerShell) — note the & prefix required
& python -m venv .venv
.venv\Scripts\activate
```

> **Windows note:** if `python -m venv .venv` gives a syntax error in PowerShell,
> use `& python -m venv .venv` or run from Command Prompt instead.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> `efficient-kan` is fetched directly from GitHub during install — internet access
> and `git` are required.  Installation takes 2–5 minutes on first run.

### 4. Train the model

`--train` and `--test` flags are **required**.

```bash
# Best result (paper configuration): train on RW9+RW10+RW11, test on RW12
python training/train.py --train RW9 RW10 RW11 --test RW12

# Single-file quick run
python training/train.py --train RW9 --test RW11

# Custom output directory
python training/train.py --train RW9 RW10 RW11 --test RW12 --out my_runs
```

> **Training time:** approximately 50 minutes per run on CPU (2.1 M samples,
> batch size 256).  No GPU is required but will speed up training if available
> — the script uses CUDA automatically if `torch.cuda.is_available()`.

Each run creates a timestamped directory under `models/KAN_{timestamp}/`:

```
models/KAN_20260315_204210/
├── best_model.pt                     PyTorch weights (best validation epoch)
├── splines_kan.json                  B-spline coefficients for NumPy inference
├── config.json                       Norm params, OCV polynomial, hyperparameters
├── results.mat                       Predictions, targets, metrics
├── log.txt                           Timestamped training log
├── training_loss_*.pdf
├── predicted_vs_actual_*.pdf
├── episode_profile_*.pdf             Single RW episode: voltage + current
├── kan_activations_*.pdf
└── error_distribution_*.pdf
```

### 5. (Optional) Rebuild dataset from raw files

The preprocessed `dataset/` files are committed — you do **not** need the raw
data to train.  If you have the original NASA PCoE RW*.mat files and want to
regenerate:

```bash
python build_dataset.py --raw path/to/raw/Matlab
```

Raw files available at:
[https://data.nasa.gov/dataset/randomized-battery-usage-1-random-walk](https://data.nasa.gov/dataset/randomized-battery-usage-1-random-walk)

---

## Dataset

**NASA PCoE Randomized Battery Usage dataset**

Cell: ePLB C020 Li-ion (2.1 Ah nominal capacity).
Profiles RW9–RW12: repeated random walk charge/discharge pulses at room temperature.
Raw sampling: ~1 s. Preprocessed: decimated to 10 s effective sample period (~770K samples per file).

Variables in each `dataset/RW*.mat`:

| Variable | Description |
|---|---|
| `V` | Terminal voltage [V] |
| `I` | Current [A], positive = discharge |
| `T` | Temperature [°C] |
| `SoC` | State of Charge [0–1], Coulomb counting anchored at 1.0 after each reference charge |
| `Q` | Cumulative charge per episode [Ah], resets at episode boundaries |
| `t` | Absolute time [s] |
| `Cn` | Nominal capacity [Ah] (~2.1 Ah, estimated from reference discharge) |
| `Ts` | Effective sample period [s] (10 s after decimation) |

---

## Model

**Architecture:** KAN with layers [4 → 16 → 1], grid_size=10, spline_order=3 (~2K parameters)

**Inputs:** `[I, SoC, T, ΔI]` — current, state of charge, temperature, current step

**Output:** Terminal voltage V [V]

### Best single run (RW9+RW10+RW11 → RW12)

| Metric | Value |
|---|---|
| RMSE | 81.1 mV |
| MAE  | 60.9 mV |
| R²   | 0.929 |

Model directory: `models/KAN_20260315_204210/`

### Leave-one-out cross-validation (all 4 profiles)

| Fold | Train | Test | RMSE | MAE | R² |
|---|---|---|---|---|---|
| CV1 | RW9+RW10+RW11 | RW12 | 81.1 mV | 60.9 mV | 0.929 |
| CV2 | RW9+RW10+RW12 | RW11 | 94.6 mV | 71.5 mV | 0.901 |
| CV3 | RW9+RW11+RW12 | RW10 | 87.9 mV | 68.9 mV | 0.914 |
| CV4 | RW10+RW11+RW12 | RW9 | 90.9 mV | 71.0 mV | 0.909 |
| **Mean** | | | **88.6 mV** | **68.1 mV** | **0.913** |
| **Std** | | | **±4.9 mV** | **±4.2 mV** | **±0.010** |

All four folds achieve R² > 0.90.

**Deployment:** B-spline coefficients exported to `splines_kan.json`.
Inference uses exact Cox-de Boor recursion in NumPy — zero approximation error,
no PyTorch required at runtime. Self-check max diff PyTorch vs NumPy: ≤ 1.79×10⁻⁷ V.

---

## HEMS integration (ESS_kan.py)

`ESS_kan.py` is a drop-in replacement for `ESS_ennc_lite.py` in the CIPAR Labs HEMS framework.

It loads `splines_kan.json` from a directory you specify:

```python
from ESS_kan import ESS_KAN

# Point to the run directory that contains splines_kan.json
ess = ESS_KAN(splines_dir="models/KAN_20260315_204210")
```

> **Important:** `splines_kan.json` lives inside the timestamped run folder produced by
> `train.py`, not in the repo root.  You must pass `splines_dir` explicitly pointing to
> the run you want to deploy.  To use a stable path, copy `splines_kan.json` and
> `config.json` to a fixed location and point `splines_dir` there.

Interface (same as ESS base class):

```python
ess.update_SoE_ch(p_GL_S, p_GL, delta_t)       # charging step  -> excess [kW]
ess.update_SoE_dch(p_GL_S, delta_t)             # discharge step -> lack   [kW]
ess.get_wear_cost(SoE_prev, p_S_k, delta_t)     # wear cost      -> C_b_k  [EUR]
```

---

## Requirements

```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
efficient-kan @ git+https://github.com/Blealtan/efficient-kan.git
```

PyTorch and efficient-kan are only needed for **training**.
Runtime inference (`ESS_kan.py`) requires **numpy only**.

---

## References

- Z. Liu et al., "KAN: Kolmogorov-Arnold Networks," arXiv:2404.19756, 2024.
- M. H. Sulaiman et al., "Battery state of charge estimation for electric vehicle using Kolmogorov-Arnold networks," *Energy* 311 (2024) 133417. DOI: 10.1016/j.energy.2024.133417
- NASA PCoE, Randomized Battery Usage Dataset. https://data.nasa.gov/dataset/randomized-battery-usage-1-random-walk
<!-- KAN ESS model for REC HEMS -->
