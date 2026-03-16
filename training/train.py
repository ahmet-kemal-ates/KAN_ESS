"""
train.py -- KAN training script for [I, SoC, T] -> V (terminal voltage).

Outputs saved to  models/KAN_{timestamp}/
    best_model.pt               PyTorch model weights
    config.json                 norm params + OCV polynomial + hyperparams
    splines_kan.json            exact B-spline coefficients for NumPy inference
    results.mat                 predictions, targets, metrics (scipy format)
    log.txt                     timestamped training log
    training_loss_{ts}.pdf      train/val loss curves
    predicted_vs_actual_{ts}.pdf voltage time series (first 2000 samples)
    episode_profile_{ts}.pdf    single RW episode: voltage + current profile
    kan_activations_{ts}.pdf    activation curves for I, SoC, T
    error_distribution_{ts}.pdf error histogram

Usage
-----
    python -m training.train                        # defaults
    python -m training.train --train RW9 --test RW11
    python -m training.train --train RW9 RW10 --test RW11 RW12
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Make sure the package root is importable regardless of cwd
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from training.data_loader import build_train_tensors, compute_norm_params, normalise
from training.kan_model import (
    build_kan,
    eval_splines_numpy,
    export_splines,
    get_activation_curves,
)
from training.preprocess import fit_ocv_polynomial, load_preprocessed_mat

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

HP = {
    "inputs":       ["I", "SoC", "T", "dI"],
    "output":       "V",
    "hidden":       16,
    "grid_size":    10,
    "spline_order": 3,
    "lr":           3e-4,
    "weight_decay": 1e-4,
    "batch_size":   256,
    "n_epochs":     500,
    "lr_patience":  30,
    "lr_factor":    0.5,
    "lr_min":       1e-6,
    "es_patience":  80,
    "headroom":     0.2,
    "val_split":    0.1,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dataset_path(name):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "dataset", f"{name}.mat")


def _log(fh, msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    fh.write(line + "\n")
    fh.flush()


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _save_loss_plot(train_losses, val_losses, out_dir, ts):
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.semilogy(epochs, train_losses, label="Train MSE")
    ax.semilogy(epochs, val_losses,   label="Val MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (normalised V)")
    ax.set_title("KAN Training Loss")
    ax.legend()
    ax.grid(True, which="both", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"training_loss_{ts}.pdf"))
    plt.close(fig)


def _save_pred_plot(y_true_V, y_pred_V, out_dir, ts):
    fig, ax = plt.subplots(figsize=(12, 4))
    n = min(2000, len(y_true_V))
    ax.plot(y_true_V[:n], label="Measured", lw=1.0)
    ax.plot(y_pred_V[:n], label="Predicted", lw=1.0, alpha=0.8)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("V [V]")
    ax.set_title("Predicted vs Measured Voltage (first 2000 samples)")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"predicted_vs_actual_{ts}.pdf"))
    plt.close(fig)


def _save_episode_plot(V_true, V_pred, I_phys, out_dir, ts, start=300, length=400):
    """Single RW episode: voltage (measured + predicted) and current, two panels."""
    end = min(start + length, len(V_true))
    idx = np.arange(end - start)
    t_s = idx * 10  # 10 s per sample -> seconds

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.plot(t_s, V_true[start:end], label="Measured", lw=1.2, color="steelblue")
    ax1.plot(t_s, V_pred[start:end], label="Predicted", lw=1.2,
             color="darkorange", alpha=0.85)
    ax1.set_ylabel("Voltage [V]")
    ax1.set_title("KAN Voltage Model — Single RW Episode")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.4)

    ax2.step(t_s, I_phys[start:end], where="post", lw=1.2, color="seagreen")
    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Current [A]")
    ax2.set_title("Applied Current (positive = discharge)")
    ax2.grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"episode_profile_{ts}.pdf"))
    plt.close(fig)


def _save_activation_plot(curves, out_dir, ts):
    labels = {"I": "Current I [A]", "SoC": "SoC [-]", "T": "Temperature [C]",
              "dI": "Current step dI [A]"}
    items = list(curves.items())
    n = len(items)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows))
    axes_flat = axes.flatten() if n > 1 else [axes]
    for ax, (name, (x, y)) in zip(axes_flat, items):
        ax.plot(x, y, lw=1.5)
        ax.set_xlabel(labels.get(name, name))
        ax.set_ylabel("V [V]")
        ax.set_title(f"KAN activation: {name} -> V")
        ax.grid(True, alpha=0.4)
    # Hide any unused subplot if odd number of curves
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"kan_activations_{ts}.pdf"))
    plt.close(fig)


def _save_error_plot(errors_mV, out_dir, ts):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors_mV, bins=80, color="steelblue", edgecolor="white", lw=0.3)
    ax.axvline(0, color="red", lw=1.5, ls="--")
    ax.set_xlabel("Error [mV]")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Error Distribution")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"error_distribution_{ts}.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(train_names, test_names, out_root="models"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_root, f"KAN_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "log.txt")
    fh = open(log_path, "w", encoding="utf-8")

    _log(fh, f"KAN ESS training  --  task: [I, SoC, T, dI] -> V")
    _log(fh, f"Train files : {train_names}")
    _log(fh, f"Test  files : {test_names}")
    _log(fh, f"Output dir  : {run_dir}")
    _log(fh, "")

    # ── Load datasets ──────────────────────────────────────────────────────
    _log(fh, "Loading preprocessed datasets...")
    train_datasets = [load_preprocessed_mat(_dataset_path(n)) for n in train_names]
    test_datasets  = [load_preprocessed_mat(_dataset_path(n)) for n in test_names]

    for ds in train_datasets + test_datasets:
        _log(fh, f"  {ds['name']}: {len(ds['V']):,} samples, "
                  f"Cn={ds['Cn']:.3f} Ah, Ts={ds['Ts']:.1f} s")

    # ── Normalisation ──────────────────────────────────────────────────────
    norm = compute_norm_params(train_datasets, test_datasets,
                               headroom=HP["headroom"])
    _log(fh, f"\nNorm params: {json.dumps({k: round(float(v), 4) for k, v in norm.items()})}")

    # ── OCV polynomial ─────────────────────────────────────────────────────
    _log(fh, "\nFitting OCV polynomial (degree-5)...")
    ocv_coefs = fit_ocv_polynomial(train_datasets + test_datasets, degree=5)
    _log(fh, f"  OCV coefs (high->low): {[round(c, 5) for c in ocv_coefs]}")

    # ── Build tensors ──────────────────────────────────────────────────────
    X_all, y_all = build_train_tensors(train_datasets, norm)

    # Validation split (last val_split fraction, chronologically)
    n_val   = max(1, int(len(X_all) * HP["val_split"]))
    n_train = len(X_all) - n_val

    X_tr = torch.from_numpy(X_all[:n_train])
    y_tr = torch.from_numpy(y_all[:n_train])
    X_va = torch.from_numpy(X_all[n_train:])
    y_va = torch.from_numpy(y_all[n_train:])

    _log(fh, f"\nTrain samples: {n_train:,}  |  Val samples: {n_val:,}")

    loader_tr = DataLoader(TensorDataset(X_tr, y_tr),
                           batch_size=HP["batch_size"], shuffle=True)

    # ── Build model ────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(fh, f"\nDevice: {device}")
    _log(fh, f"HP: {json.dumps(HP)}")

    model = build_kan(n_input=4, hidden=HP["hidden"],
                      grid_size=HP["grid_size"],
                      spline_order=HP["spline_order"],
                      device=device)

    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=HP["lr"],
                                  weight_decay=HP["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=HP["lr_factor"],
        patience=HP["lr_patience"], min_lr=HP["lr_min"]
    )

    # ── Training loop ──────────────────────────────────────────────────────
    best_val   = float("inf")
    best_epoch = 0
    es_count   = 0
    train_losses, val_losses = [], []
    best_path = os.path.join(run_dir, "best_model.pt")

    _log(fh, f"\n{'Epoch':>6}  {'Train MSE':>12}  {'Val MSE':>12}  {'LR':>10}")
    _log(fh, "-" * 48)

    t0 = time.time()
    for epoch in range(1, HP["n_epochs"] + 1):
        # Train
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader_tr:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * len(xb)
        train_mse = epoch_loss / n_train

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_va.to(device))
            val_mse  = criterion(val_pred, y_va.to(device)).item()

        scheduler.step(val_mse)
        train_losses.append(train_mse)
        val_losses.append(val_mse)

        lr_now = optimiser.param_groups[0]["lr"]

        if val_mse < best_val:
            best_val   = val_mse
            best_epoch = epoch
            es_count   = 0
            torch.save(model.state_dict(), best_path)
        else:
            es_count += 1

        if epoch % 20 == 0 or epoch == 1:
            _log(fh, f"{epoch:>6}  {train_mse:>12.6f}  {val_mse:>12.6f}  {lr_now:>10.2e}")

        if es_count >= HP["es_patience"]:
            _log(fh, f"\nEarly stopping at epoch {epoch} (no improvement for {HP['es_patience']} epochs)")
            break

    elapsed = time.time() - t0
    _log(fh, f"\nTraining done in {elapsed:.1f} s.  Best epoch: {best_epoch}  Best val MSE: {best_val:.6f}")

    # ── Load best model ────────────────────────────────────────────────────
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    # ── Test evaluation ────────────────────────────────────────────────────
    _log(fh, "\n--- Test Evaluation ---")
    V_range = norm["V_max"] - norm["V_min"]

    all_true_V, all_pred_V, all_I = [], [], []
    for ds in test_datasets:
        X_te, y_te = normalise(ds, norm)
        with torch.no_grad():
            y_hat_n = model(torch.from_numpy(X_te).to(device)).cpu().numpy()

        V_true = y_te.squeeze()    * V_range + norm["V_min"]
        V_pred = y_hat_n.squeeze() * V_range + norm["V_min"]
        all_true_V.append(V_true)
        all_pred_V.append(V_pred)
        all_I.append(ds["I"])

        err_mV = (V_pred - V_true) * 1000.0
        rmse_mV = float(np.sqrt(np.mean(err_mV ** 2)))
        mae_mV  = float(np.mean(np.abs(err_mV)))
        ss_res  = np.sum((V_true - V_pred) ** 2)
        ss_tot  = np.sum((V_true - V_true.mean()) ** 2)
        r2      = float(1.0 - ss_res / (ss_tot + 1e-15))
        _log(fh, f"  {ds['name']}: RMSE={rmse_mV:.2f} mV  MAE={mae_mV:.2f} mV  R2={r2:.4f}")

    y_true_V = np.concatenate(all_true_V)
    y_pred_V = np.concatenate(all_pred_V)
    I_all    = np.concatenate(all_I)
    err_mV_all = (y_pred_V - y_true_V) * 1000.0

    rmse_tot = float(np.sqrt(np.mean(err_mV_all ** 2)))
    mae_tot  = float(np.mean(np.abs(err_mV_all)))
    ss_res   = np.sum((y_true_V - y_pred_V) ** 2)
    ss_tot   = np.sum((y_true_V - y_true_V.mean()) ** 2)
    r2_tot   = float(1.0 - ss_res / (ss_tot + 1e-15))
    _log(fh, f"\n  OVERALL: RMSE={rmse_tot:.2f} mV  MAE={mae_tot:.2f} mV  R2={r2_tot:.4f}")

    # ── Spline export ──────────────────────────────────────────────────────
    _log(fh, "\nExporting B-spline coefficients...")
    layers_data = export_splines(model)

    # Self-check: PyTorch vs NumPy max diff
    X_check = X_te[:500]
    with torch.no_grad():
        pt_out = model(torch.from_numpy(X_check).to(device)).cpu().numpy().squeeze()
    np_out = eval_splines_numpy(layers_data, X_check).squeeze()
    max_diff = float(np.max(np.abs(pt_out - np_out)))
    _log(fh, f"  PyTorch vs NumPy spline max diff: {max_diff:.2e}  (should be <1e-5)")

    splines_path = os.path.join(run_dir, "splines_kan.json")
    splines_json = {
        "layers":    layers_data,
        "norm":      {k: float(v) for k, v in norm.items()},
        "ocv_coefs": ocv_coefs,
        "hp":        HP,
    }
    with open(splines_path, "w") as f:
        json.dump(splines_json, f, indent=2)
    _log(fh, f"  Saved: {splines_path}")

    # ── config.json ────────────────────────────────────────────────────────
    config = {
        "task":       {"inputs": HP["inputs"], "output": HP["output"]},
        "norm":       {k: float(v) for k, v in norm.items()},
        "ocv_polynomial": {"degree": 5, "coefs_high_to_low": ocv_coefs},
        "hp":         HP,
        "best_epoch": best_epoch,
        "metrics": {
            "rmse_mV": rmse_tot,
            "mae_mV":  mae_tot,
            "r2":      r2_tot,
        },
        "train_files": train_names,
        "test_files":  test_names,
        "timestamp":   ts,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ── results.mat ────────────────────────────────────────────────────────
    scipy.io.savemat(os.path.join(run_dir, "results.mat"), {
        "V_true_V":   y_true_V.astype(np.float32),
        "V_pred_V":   y_pred_V.astype(np.float32),
        "err_mV":     err_mV_all.astype(np.float32),
        "rmse_mV":    np.float32(rmse_tot),
        "mae_mV":     np.float32(mae_tot),
        "r2":         np.float32(r2_tot),
        "train_loss": np.array(train_losses, dtype=np.float32),
        "val_loss":   np.array(val_losses,   dtype=np.float32),
    })

    # ── Plots ──────────────────────────────────────────────────────────────
    _log(fh, "\nGenerating plots...")
    _save_loss_plot(train_losses, val_losses, run_dir, ts)
    _save_pred_plot(y_true_V, y_pred_V, run_dir, ts)
    _save_episode_plot(y_true_V, y_pred_V, I_all, run_dir, ts)
    _save_error_plot(err_mV_all, run_dir, ts)

    try:
        curves = get_activation_curves(model, norm, n_points=200, device=device)
        _save_activation_plot(curves, run_dir, ts)
    except Exception as e:
        _log(fh, f"  [warn] Activation plot skipped: {e}")

    _log(fh, f"\nAll outputs saved to: {run_dir}")
    fh.close()
    return run_dir


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train KAN ESS voltage model",
        epilog=(
            "Example (paper result):\n"
            "  python training/train.py --train RW9 RW10 RW11 --test RW12\n\n"
            "Available datasets: RW9  RW10  RW11  RW12  (must exist in dataset/)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train", nargs="+", required=True,
                        metavar="RW", help="Training dataset names (e.g. RW9 RW10 RW11)")
    parser.add_argument("--test",  nargs="+", required=True,
                        metavar="RW", help="Test dataset names (e.g. RW12)")
    parser.add_argument("--out",   default="models",
                        help="Root output directory (default: models)")
    args = parser.parse_args()

    train(train_names=args.train, test_names=args.test, out_root=args.out)
