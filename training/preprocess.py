"""
preprocess.py — ePLB C020 Random Walk dataset loader.

Parses raw RW*.mat files (step-by-step struct format) and extracts:
  - Random walk sections only (charge/discharge/rest tagged 'random walk')
  - SoC via vectorised Coulomb counting, anchored at 1.0 after each
    reference charge
  - Q_conducted: cumulative charge passed since episode start [Ah/Cn],
    reset at each episode (between reference charges)
  - Cn estimated from the first reference discharge in each file

Returns a dict per file:  V, I, T, t, SoC, Q  (all float32 numpy arrays)
plus scalars  Ts, Cn, name.

Usage
-----
    from training.preprocess import load_rw_file
    data = load_rw_file("path/to/RW9.mat")
    # data["V"], data["I"], data["SoC"], data["Q"], ...
"""

import os
import numpy as np
import scipy.io


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _squeeze_str(x):
    return str(np.asarray(x).squeeze())


def _col(x):
    return np.asarray(x).squeeze().astype(np.float64)


def _estimate_cn(step, n_steps):
    """Estimate Cn [Ah] from the first reference discharge step."""
    for i in range(min(20, n_steps)):
        comment = _squeeze_str(step[0, i]["comment"])
        if "reference discharge" in comment.lower():
            t = _col(step[0, i]["time"])
            I = _col(step[0, i]["current"])
            if t.ndim == 0 or t.size < 2:
                continue
            dt = np.diff(t, prepend=t[0])
            dt[0] = dt[1] if dt.size > 1 else 1.0
            # discharge: positive current convention → Cn = sum(I * dt) / 3600
            cn = float(np.sum(np.abs(I) * dt) / 3600.0)
            if cn > 0.5:          # sanity: must be > 0.5 Ah
                return cn
    return 2.1                    # fallback for ePLB C020


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_rw_file(mat_path):
    """
    Load one RW*.mat file and return the Random Walk measurement section.

    Parameters
    ----------
    mat_path : str
        Path to an RW*.mat file from the ePLB dataset.

    Returns
    -------
    dict with keys:
        name  : filename (str)
        path  : full path (str)
        V     : terminal voltage  [N]  float32  [V]
        I     : current           [N]  float32  [A], positive = discharge
        T     : temperature       [N]  float32  [°C]
        t     : time              [N]  float32  [s], continuous
        SoC   : state of charge   [N]  float32  [0–1]
        Q     : conducted charge  [N]  float32  [Ah/Cn], resets each episode
        Ts    : sampling interval float  [s]
        Cn    : nominal capacity  float  [Ah]
    """
    d = scipy.io.loadmat(mat_path)
    step = d["data"]["step"][0, 0]
    n_steps = step.shape[1]

    cn = _estimate_cn(step, n_steps)
    Ts = 1.0            # 1 s uniform sampling (confirmed from data)

    V_segs, I_segs, T_segs, t_segs, SoC_segs, Q_segs = [], [], [], [], [], []
    t_offset = 0.0
    current_SoC = None   # anchored to 1.0 after each reference charge
    episode_Q   = 0.0    # cumulative charge within current RW episode [Ah]

    for i in range(n_steps):
        s = step[0, i]
        comment = _squeeze_str(s["comment"])
        t_arr = _col(s["time"])

        if t_arr.ndim == 0 or t_arr.size < 2:
            continue

        V_arr = _col(s["voltage"])
        I_arr = _col(s["current"])
        T_arr = _col(s["temperature"])

        # dt per sample (forward difference, first sample = second interval)
        dt = np.diff(t_arr, prepend=t_arr[0])
        dt[0] = dt[1] if dt.size > 1 else Ts

        comment_lo = comment.lower()

        # ----------------------------------------------------------------
        # Reference charge: battery is brought to 100 % SoC
        # ----------------------------------------------------------------
        if "reference charge" in comment_lo:
            current_SoC = 1.0
            episode_Q   = 0.0
            continue

        # Reference discharge: update running SoC but skip from dataset
        if "reference discharge" in comment_lo:
            if current_SoC is not None:
                dq = float(np.sum(np.abs(I_arr) * dt) / 3600.0)
                current_SoC = float(np.clip(current_SoC - dq / cn, 0.0, 1.0))
            episode_Q = 0.0
            continue

        # Non-RW steps (pulsed load, etc.): track SoC but skip
        if "random walk" not in comment_lo:
            if current_SoC is not None:
                # signed: positive I = discharge → decreases SoC
                dq = float(np.sum(I_arr * dt) / 3600.0)
                current_SoC = float(np.clip(current_SoC - dq / cn, 0.0, 1.0))
            continue

        # Skip if SoC anchor not yet established
        if current_SoC is None:
            continue

        # ----------------------------------------------------------------
        # Random walk step — include in dataset
        # ----------------------------------------------------------------
        # Continuous time axis
        t_rel = (t_arr - t_arr[0]) + t_offset
        t_offset = float(t_rel[-1]) + Ts

        # Vectorised Coulomb counting within step
        # delta_ah[k] = charge discharged during sample k  (+ = discharge)
        delta_ah = I_arr * dt / 3600.0
        # Q_conducted: cumulative from episode start (normalised by Cn)
        q_step = episode_Q + np.cumsum(np.concatenate([[0.0], delta_ah[:-1]])) / cn

        # SoC at each sample: anchored from start of step
        soc_at_start = current_SoC
        soc_step = soc_at_start - (np.cumsum(np.concatenate([[0.0], delta_ah[:-1]])) / cn)
        soc_step = np.clip(soc_step, 0.0, 1.0)

        V_segs.append(V_arr)
        I_segs.append(I_arr)
        T_segs.append(T_arr)
        t_segs.append(t_rel)
        SoC_segs.append(soc_step)
        Q_segs.append(q_step)

        # Update running state
        current_SoC = float(np.clip(soc_step[-1], 0.0, 1.0))
        episode_Q   = float(q_step[-1]) + float(delta_ah[-1]) / cn

    if not V_segs:
        raise ValueError(f"No Random Walk data found in {mat_path}")

    return {
        "name": os.path.basename(mat_path),
        "path": mat_path,
        "V":   np.concatenate(V_segs).astype(np.float32),
        "I":   np.concatenate(I_segs).astype(np.float32),
        "T":   np.concatenate(T_segs).astype(np.float32),
        "t":   np.concatenate(t_segs).astype(np.float32),
        "SoC": np.concatenate(SoC_segs).astype(np.float32),
        "Q":   np.concatenate(Q_segs).astype(np.float32),
        "Ts":  Ts,
        "Cn":  cn,
    }


def load_preprocessed_mat(mat_path):
    """
    Load a preprocessed dataset/RW*.mat file produced by build_dataset.py.

    Returns the same dict format as load_rw_file() so the rest of the
    pipeline (data_loader, train.py) works identically.
    """
    d = scipy.io.loadmat(mat_path)
    return {
        "name": os.path.basename(mat_path),
        "path": mat_path,
        "V":   np.asarray(d["V"]).squeeze().astype(np.float32),
        "I":   np.asarray(d["I"]).squeeze().astype(np.float32),
        "T":   np.asarray(d["T"]).squeeze().astype(np.float32),
        "t":   np.asarray(d["t"]).squeeze().astype(np.float32),
        "SoC": np.asarray(d["SoC"]).squeeze().astype(np.float32),
        "Q":   np.asarray(d["Q"]).squeeze().astype(np.float32),
        "Ts":  float(np.asarray(d["Ts"]).squeeze()),
        "Cn":  float(np.asarray(d["Cn"]).squeeze()),
    }


def fit_ocv_polynomial(datasets, degree=5):
    """
    Fit an OCV–SoC polynomial  V = poly(SoC)  from rest steps.

    Uses all samples where |I| < 0.05 A across the provided datasets.
    Returns numpy polynomial coefficients (highest power first), suitable
    for np.polyval(coefs, SoC).
    """
    V_rest, S_rest = [], []
    for ds in datasets:
        mask = np.abs(ds["I"]) < 0.05
        if mask.sum() > 10:
            V_rest.append(ds["V"][mask])
            S_rest.append(ds["SoC"][mask])

    if not V_rest:
        raise RuntimeError("No near-rest samples found for OCV fitting.")

    V_all = np.concatenate(V_rest)
    S_all = np.concatenate(S_rest)
    coefs = np.polyfit(S_all, V_all, degree)
    return coefs.tolist()
