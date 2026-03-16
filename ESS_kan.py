"""
ESS_kan.py -- KAN-based ESS model for the CIPAR Labs REC HEMS framework.

Drop-in replacement for ESS_ennc_lite.py.  Requires one file produced by
training/train.py:
    splines_kan.json    B-spline layers + norm params + OCV polynomial

Runtime dependencies: numpy only (no PyTorch at inference time).

Inference:  [I_norm, SoC, T_norm]  ->  V_norm  ->  V [V]
            using exact B-spline evaluation (zero approximation error).

Interface (same as base ESS class):
    update_SoE_ch(p_GL_S, p_GL, delta_t)   -> excess [kW]
    update_SoE_dch(p_GL_S, delta_t)        -> lack   [kW]
    get_wear_cost(SoE_prev, p_S_k, delta_t) -> C_b_k  [EUR]
"""

import os
import json
import numpy as np


# ---------------------------------------------------------------------------
# B-spline NumPy evaluation  (mirrors training/kan_model.py exactly)
# ---------------------------------------------------------------------------

def _b_splines_numpy(x, grid, spline_order):
    """
    Cox-de Boor recursion.

    x    : (batch, in_features)
    grid : (in_features, n_knots)
    Returns (batch, in_features, grid_size + spline_order)
    """
    x_e   = x[:, :, np.newaxis]
    bases = ((x_e >= grid[:, :-1]) & (x_e < grid[:, 1:])).astype(np.float32)
    for k in range(1, spline_order + 1):
        d1 = grid[:, k:-1]   - grid[:, :-(k + 1)]
        d2 = grid[:, k + 1:] - grid[:, 1:(-k)]
        d1 = np.where(np.abs(d1) > 1e-8, d1, 1.0)
        d2 = np.where(np.abs(d2) > 1e-8, d2, 1.0)
        bases = (
            (x_e - grid[:, :-(k + 1)]) / d1 * bases[:, :, :-1]
            + (grid[:, k + 1:] - x_e)  / d2 * bases[:, :, 1:]
        )
    return bases


def _eval_splines_numpy(layers_data, x):
    """
    Pure-NumPy forward pass through an exported KAN.

    layers_data : list of dicts from export_splines()
    x           : (batch, in_features) float32
    Returns     : (batch, out_features) float32
    """
    for ld in layers_data:
        grid  = np.array(ld["grid"],                 dtype=np.float32)
        ssw   = np.array(ld["scaled_spline_weight"], dtype=np.float32)
        bw    = np.array(ld["base_weight"],          dtype=np.float32)
        order = int(ld["spline_order"])

        silu     = x / (1.0 + np.exp(-x.astype(np.float64))).astype(np.float32)
        base_out = silu @ bw.T

        B, F       = x.shape
        bases_flat = _b_splines_numpy(x, grid, order).reshape(B, -1)
        ssw_flat   = ssw.reshape(ssw.shape[0], -1)
        spline_out = bases_flat @ ssw_flat.T

        x = base_out + spline_out
    return x


# ---------------------------------------------------------------------------
# ESS base class (inline minimal copy -- replace with framework import)
# ---------------------------------------------------------------------------
# from EMS.Community.MG.ESS.ESS import ESS   # <- use inside CIPAR framework

class ESS:
    """Minimal ESS base -- replace with framework import when integrating."""
    def __init__(self, model, Q, p_S_max, a, b, B, eta, SoE_0,
                 V_n, SoE_min, SoE_max, Q_n):
        self.model   = model
        self.Q       = Q
        self.p_S_max = p_S_max
        self.a       = a
        self.b       = b
        self.B       = B
        self.eta     = eta
        self.SoE_0   = SoE_0
        self.V_n     = V_n
        self.SoE_min = SoE_min
        self.SoE_max = SoE_max
        self.Q_n     = Q_n
        self.SoE     = SoE_0


# ---------------------------------------------------------------------------
# ESS_kan
# ---------------------------------------------------------------------------

class ESS_kan(ESS):
    """
    KAN-based ESS model.  Uses exact B-spline inference from splines_kan.json.

    Parameters
    ----------
    splines_dir : str
        Directory containing splines_kan.json.  Defaults to directory of
        this file.
    All other parameters are forwarded to the ESS base class.
    """

    def __init__(self, model="KAN", Q=5.0, p_S_max=7.0,
                 a=694.0, b=0.795, B=0.0, eta=0.98,
                 SoE_0=0.5, V_n=3.6, SoE_min=0.15, SoE_max=0.95,
                 Q_n=None, splines_dir=None):

        if splines_dir is None:
            splines_dir = os.path.dirname(os.path.abspath(__file__))

        if Q_n is None:
            Q_n = (Q * 1e3) / V_n   # Ah

        super().__init__(model, Q, p_S_max, a, b, B, eta,
                         SoE_0, V_n, SoE_min, SoE_max, Q_n)

        # ── Load splines_kan.json ──────────────────────────────────────────
        splines_path = os.path.join(splines_dir, "splines_kan.json")
        with open(splines_path, "r") as f:
            data = json.load(f)

        self._layers    = data["layers"]
        norm            = data["norm"]
        self._norm      = norm
        self._ocv_coefs = np.array(data["ocv_coefs"], dtype=np.float64)

        # Normalisation shorthands
        self._V_min     = float(norm["V_min"])
        self._V_max     = float(norm["V_max"])
        self._I_abs_max = float(norm["I_abs_max"])
        self._T_min     = float(norm["T_min"])
        self._T_max     = float(norm["T_max"])

    # ── Internal helpers ───────────────────────────────────────────────────

    def _ocv(self, soc):
        """OCV-SoC polynomial estimate [V]."""
        return float(np.polyval(self._ocv_coefs, np.clip(soc, 0.0, 1.0)))

    def _predict_V(self, I_A, soc, T_C=25.0):
        """
        Predict terminal voltage [V] from physical inputs.

        Parameters
        ----------
        I_A  : float  current [A], positive = discharge
        soc  : float  state-of-charge [0, 1]
        T_C  : float  temperature [degC]  (default 25 degC)
        """
        I_n = np.float32(I_A / (self._I_abs_max + 1e-15))
        S_n = np.float32(np.clip(soc, 0.0, 1.0))
        T_n = np.float32(
            (T_C - self._T_min) / (self._T_max - self._T_min + 1e-15)
        )
        x   = np.array([[I_n, S_n, T_n]], dtype=np.float32)
        V_n = float(_eval_splines_numpy(self._layers, x)[0, 0])
        V   = V_n * (self._V_max - self._V_min) + self._V_min
        return float(np.clip(V, self._V_min, self._V_max))

    def _current_from_power(self, p_kW, soc, T_C=25.0):
        """
        Approximate current [A] from power [kW].

        Uses KAN-predicted voltage (more accurate than OCV polynomial alone).
        Falls back to OCV if prediction fails.
        """
        # Initial estimate with OCV
        V_est = self._ocv(soc)
        if abs(V_est) < 0.1:
            V_est = self.V_n
        I_est = (p_kW * 1e3) / V_est

        # One refinement with KAN prediction
        try:
            V_kan = self._predict_V(I_est, soc, T_C)
            if abs(V_kan) > 0.1:
                I_est = (p_kW * 1e3) / V_kan
        except Exception:
            pass

        return I_est

    def _coulomb_update(self, I_A, delta_t):
        """
        Update SoE via Coulomb counting.

        I_A     : float  current [A], positive = discharge
        delta_t : float  time step [h]
        """
        delta_Ah = I_A * delta_t                        # [Ah]
        delta_SoE = delta_Ah / (self.Q_n + 1e-15)       # Q_n [Ah]
        self.SoE = float(np.clip(self.SoE - delta_SoE,
                                 self.SoE_min, self.SoE_max))

    # ── ESS interface ──────────────────────────────────────────────────────

    def update_SoE_ch(self, p_GL_S, p_GL, delta_t):
        """
        Charging update.

        Parameters
        ----------
        p_GL_S  : float  scheduled charge power (positive, [kW])
        p_GL    : float  available grid/local power [kW]
        delta_t : float  time step [h]

        Returns
        -------
        excess  : float  power that could not be stored [kW]
        """
        e_max = (self.SoE_max - self.SoE) * self.Q
        p_max = min(self.p_S_max, e_max / (delta_t + 1e-15))
        p_ch  = float(np.clip(min(p_GL_S, p_GL, p_max), 0.0, None))

        if p_ch > 1e-6:
            I_A = -self._current_from_power(p_ch, self.SoE)  # negative = charge
            self._coulomb_update(I_A, delta_t)

        return float(max(p_GL_S - p_ch, 0.0))

    def update_SoE_dch(self, p_GL_S, delta_t):
        """
        Discharging update.

        Parameters
        ----------
        p_GL_S  : float  scheduled discharge power (positive, [kW])
        delta_t : float  time step [h]

        Returns
        -------
        lack    : float  power deficit [kW]
        """
        e_avail = (self.SoE - self.SoE_min) * self.Q
        p_max   = min(self.p_S_max, e_avail / (delta_t + 1e-15))
        p_dch   = float(np.clip(min(p_GL_S, p_max), 0.0, None))

        if p_dch > 1e-6:
            I_A = self._current_from_power(p_dch, self.SoE)  # positive = discharge
            self._coulomb_update(I_A, delta_t)

        return float(max(p_GL_S - p_dch, 0.0))

    def get_wear_cost(self, SoE_prev, p_S_k, delta_t):
        """
        ACC-DoD battery wear cost for one time slot.

        C_b_k = B * (|delta_SoE| * Q) / (0.5 * ACC(DoD))
        ACC(DoD) = a * DoD^(-b)

        Parameters
        ----------
        SoE_prev : float  SoE at start of slot
        p_S_k    : float  power this slot [kW]
        delta_t  : float  slot duration [h]

        Returns
        -------
        C_b_k : float  wear cost [EUR]
        """
        delta_SoE = abs(self.SoE - SoE_prev)
        if delta_SoE < 1e-9:
            return 0.0
        DoD   = delta_SoE
        acc   = self.a * (DoD ** (-self.b))
        e_k   = delta_SoE * self.Q
        return float(self.B * e_k / (0.5 * acc))
