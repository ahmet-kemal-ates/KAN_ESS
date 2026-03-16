"""
data_loader.py -- Normalisation and tensor preparation for KAN training.

Task:  [I_norm, SoC, T_norm, dI_norm]  ->  V_norm
  I   -> [-1, 1]  i / I_abs_max          (with headroom)
  SoC ->  [0, 1]  unchanged              (already physical range)
  T   ->  [0, 1]  (t - T_min) / range   (with headroom)
  dI  -> [-1, 1]  diff(I) / I_abs_max   (current step, captures transients)
  V   ->  [0, 1]  (v - V_min) / range   (with headroom)  -- OUTPUT

NormParams dict is saved to config.json and loaded by ESS_kan.py at runtime.
"""

import numpy as np


def compute_norm_params(train_datasets, test_datasets, headroom=0.2):
    all_ds = train_datasets + test_datasets

    v_min = min(float(ds["V"].min()) for ds in all_ds)
    v_max = max(float(ds["V"].max()) for ds in all_ds)
    i_abs = max(float(np.abs(ds["I"]).max()) for ds in all_ds)
    t_min = min(float(ds["T"].min()) for ds in all_ds)
    t_max = max(float(ds["T"].max()) for ds in all_ds)

    def _expand(lo, hi):
        r = hi - lo
        return lo - headroom * r / 2.0, hi + headroom * r / 2.0

    v_min, v_max = _expand(v_min, v_max)
    i_abs *= (1.0 + headroom)
    t_min, t_max = _expand(t_min, t_max)

    return {
        "V_min":     v_min,
        "V_max":     v_max,
        "I_abs_max": i_abs,
        "T_min":     t_min,
        "T_max":     t_max,
        "headroom":  headroom,
    }


def normalise(ds, norm):
    """
    Returns X (N, 4) = [I_norm, SoC, T_norm, dI_norm]  and  y (N, 1) = V_norm.
    dI = diff(I), normalised by I_abs_max. Captures current steps (IR transients).
    """
    I   = ds["I"]
    I_n = I / norm["I_abs_max"]
    dI_n = np.concatenate([[0.0], np.diff(I)]) / norm["I_abs_max"]
    SoC = ds["SoC"]                                                  # [0, 1]
    T_n = (ds["T"] - norm["T_min"]) / (norm["T_max"] - norm["T_min"])
    V_n = (ds["V"] - norm["V_min"]) / (norm["V_max"] - norm["V_min"])

    X = np.stack([I_n, SoC, T_n, dI_n], axis=1).astype(np.float32)
    y = V_n.reshape(-1, 1).astype(np.float32)
    return X, y


def build_train_tensors(train_datasets, norm):
    Xs, ys = [], []
    for ds in train_datasets:
        X, y = normalise(ds, norm)
        Xs.append(X)
        ys.append(y)
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)
