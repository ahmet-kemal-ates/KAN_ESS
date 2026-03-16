"""
kan_model.py -- KAN model definition + spline export for ESS deployment.

Architecture:  KAN([4, hidden, 1])
  Inputs  (4) : I_norm, SoC, T_norm, dI_norm
  Output  (1) : V_norm   (terminal voltage, normalised)

Spline export (for ESS_kan.py runtime)
---------------------------------------
A KAN is literally a composition of B-spline activations.  Instead of
discretising to a lookup table (which introduces interpolation error),
we export the spline grid and coefficient tensors as JSON and re-evaluate
them exactly in NumPy at runtime -- zero approximation error, no PyTorch
dependency, smaller file than a LUT.

efficient-kan reference:
  https://github.com/Blealtan/efficient-kan
"""

import numpy as np
import torch
from efficient_kan import KAN


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_kan(n_input=3, hidden=8, grid_size=5, spline_order=3, device="cpu"):
    model = KAN(
        layers_hidden=[n_input, hidden, 1],
        grid_size=grid_size,
        spline_order=spline_order,
    )
    return model.to(device)


# ---------------------------------------------------------------------------
# Spline export  (replaces LUT)
# ---------------------------------------------------------------------------

def export_splines(model):
    """
    Export all KAN layer weights as plain Python lists (JSON-serialisable).

    Returns a list of dicts, one per KANLinear layer:
        grid                : (in_features, n_knots)
        scaled_spline_weight: (out_features, in_features, n_coefs)
        base_weight         : (out_features, in_features)
        spline_order        : int
    """
    layers_data = []
    for layer in model.layers:
        sw = layer.spline_weight.detach().cpu()
        if hasattr(layer, "spline_scaler"):
            sw = sw * layer.spline_scaler.detach().cpu().unsqueeze(-1)
        layers_data.append({
            "grid":                 layer.grid.detach().cpu().numpy().tolist(),
            "scaled_spline_weight": sw.numpy().tolist(),
            "base_weight":          layer.base_weight.detach().cpu().numpy().tolist(),
            "spline_order":         layer.spline_order,
        })
    return layers_data


def _b_splines_numpy(x, grid, spline_order):
    """
    Replicate efficient-kan's B-spline basis evaluation in NumPy.

    x    : (batch, in_features)
    grid : (in_features, n_knots)   -- n_knots = grid_size + 2*order + 1
    Returns (batch, in_features, grid_size + spline_order)
    """
    x_e = x[:, :, np.newaxis]                                      # (B, F, 1)
    bases = ((x_e >= grid[:, :-1]) & (x_e < grid[:, 1:])).astype(np.float32)

    for k in range(1, spline_order + 1):
        d1 = grid[:, k:-1]    - grid[:, :-(k + 1)]
        d2 = grid[:, k + 1:]  - grid[:, 1:(-k)]
        d1 = np.where(np.abs(d1) > 1e-8, d1, 1.0)
        d2 = np.where(np.abs(d2) > 1e-8, d2, 1.0)
        bases = (
            (x_e - grid[:, :-(k + 1)]) / d1 * bases[:, :, :-1]
            + (grid[:, k + 1:] - x_e)  / d2 * bases[:, :, 1:]
        )
    return bases                                                    # (B, F, C)


def eval_splines_numpy(layers_data, x):
    """
    Pure-NumPy forward pass through an exported KAN.

    layers_data : list of dicts from export_splines()
    x           : (batch, in_features) float32 ndarray

    Returns     : (batch, out_features) float32 ndarray
    """
    for ld in layers_data:
        grid  = np.array(ld["grid"],                 dtype=np.float32)  # (F, K)
        ssw   = np.array(ld["scaled_spline_weight"], dtype=np.float32)  # (O, F, C)
        bw    = np.array(ld["base_weight"],          dtype=np.float32)  # (O, F)
        order = int(ld["spline_order"])

        # SiLU base activation
        silu   = x / (1.0 + np.exp(-x.astype(np.float64))).astype(np.float32)
        base_out = silu @ bw.T                                          # (B, O)

        # Spline activation
        B, F   = x.shape
        bases  = _b_splines_numpy(x, grid, order)                       # (B, F, C)
        bases_flat = bases.reshape(B, -1)                               # (B, F*C)
        ssw_flat   = ssw.reshape(ssw.shape[0], -1)                      # (O, F*C)
        spline_out = bases_flat @ ssw_flat.T                            # (B, O)

        x = base_out + spline_out

    return x


# ---------------------------------------------------------------------------
# Activation sensitivity curves  (for explainability plots)
# ---------------------------------------------------------------------------

def get_activation_curves(model, norm, n_points=200, device="cpu"):
    """
    Sweep each input independently (others fixed at 0) and record model output.
    Returns physical-unit x values for interpretability.

        { "I":   (x_phys, y_pred),
          "SoC": (x_phys, y_pred),
          "T":   (x_phys, y_pred),
          "dI":  (x_phys, y_pred) }
    """
    model.eval()
    # [I_norm, SoC, T_norm, dI_norm]
    feature_names   = ["I", "SoC", "T", "dI"]
    physical_ranges = {
        "I":   (-norm["I_abs_max"], norm["I_abs_max"]),
        "SoC": (0.0, 1.0),
        "T":   (norm["T_min"],      norm["T_max"]),
        "dI":  (-norm["I_abs_max"], norm["I_abs_max"]),
    }
    norm_ranges = {
        "I":   (-1.0, 1.0),
        "SoC": (0.0,  1.0),
        "T":   (0.0,  1.0),
        "dI":  (-1.0, 1.0),
    }

    curves = {}
    with torch.no_grad():
        for k, name in enumerate(feature_names):
            lo, hi  = norm_ranges[name]
            x_norm  = torch.linspace(lo, hi, n_points, device=device)
            x_batch = torch.zeros(n_points, len(feature_names), device=device)
            x_batch[:, k] = x_norm

            y = model(x_batch).squeeze().cpu().numpy()
            V_range = norm["V_max"] - norm["V_min"]
            y_phys  = y * V_range + norm["V_min"]

            x_phys  = torch.linspace(*physical_ranges[name], n_points).numpy()
            curves[name] = (x_phys, y_phys)

    return curves


def get_spline_weights_summary(model):
    """L1 norm of spline weights per layer (for logging)."""
    return [
        layer.spline_weight.detach().cpu().abs().mean(dim=-1).numpy()
        for layer in model.layers
    ]
