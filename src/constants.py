"""Verified manuscript values (REAL-NOW data for the figures).

Every number here is a value the manuscript reports exactly; figure panels marked
REAL-NOW render straight from these constants. Do NOT edit to "make a panel look
better" — change the manuscript first. NEEDS-DATA panels load distributions from
data/in/ instead (see data_io.py).
"""
from __future__ import annotations
import math
import numpy as np

# ----------------------------------------------------------------------------
# Calibration / conformal scalars
# ----------------------------------------------------------------------------
ALPHA = 1e-3
SIC_SATURATION = 1.0 / math.sqrt(ALPHA)   # = 31.62...
ACHIEVED_EPSB_M = 9e-4                     # achieved background FPR of M at alpha=1e-3
PER_BIN_FLOOR = 1.67e-4                    # 1/(|C1,b|+1)
C1_PER_BIN = 6000

CALIB_PVAL_MEAN = 0.4997
CALIB_TAIL_BELOW_1EM3 = 1.1e-3

# Per-region chi^2 null check: measured mean vs expected |G| (the 3 layers)
CHI2_LAYERS = {
    # name: (measured_mean, expected_|G|)
    "layer 0": (286.5, 288),
    "layer 1": (143.3, 144),
    "layer 2": (74.0, 72),
}
CORE_PHOTON_SG = {"mean": -0.05, "std": 1.06}   # layer-1 core photon s_G

# Flow-II latent norm per seed (on shell)
ZII_NORM_PER_SEED = [503.85, 505.98, 495.61]
D_II = 504

# ----------------------------------------------------------------------------
# Dead-zone Table 1 (5 GeV; three-seed median[min, max])
# Each entry: (median, min, max)
# ----------------------------------------------------------------------------
DEADZONE = {
    "z=1.00": {
        "regime": "dead zone",
        "T_raw": (0.07, 0.05, 0.83),
        "global": (12.1, 11.7, 14.4),
        "M": (28.65, 25.62, 28.78),
        "supervised": (30.96, 30.94, 30.98),
        "rho": (0.92, 0.83, 0.93),
    },
    "z=1.04": {
        "regime": "dead zone",
        "T_raw": (1.26, 0.95, 3.00),
        "global": (16.2, 10.5, 20.6),
        "M": (30.13, 28.76, 30.24),
        "supervised": (31.01, 30.97, 31.06),
        "rho": (0.97, 0.93, 0.97),
    },
    "z=0.33": {
        "regime": "resolved",
        "T_raw": (27.10, 25.45, 27.18),
        "global": (29.8, 28.4, 29.9),
        "M": (31.03, 29.82, 31.56),
        "supervised": (31.28, 31.27, 31.28),
        "rho": (0.94, 0.75, 1.07),
    },
    "z=1.16": {
        "regime": "intermediate",
        "T_raw": (13.61, 11.16, 21.61),
        "global": (12.7, 6.0, 15.5),
        "M": (28.72, 28.13, 29.89),
        "supervised": (30.94, 30.79, 31.05),
        "rho": (0.88, 0.85, 0.89),
    },
}
# Panel (b) bar order
DEADZONE_BAR_CELLS = ["z=1.00", "z=1.04", "z=0.33", "z=1.16"]
SCORE_KEYS = ["T_raw", "global", "M", "supervised"]


def med_err(triple):
    """(median, min, max) -> (median, [lower_err, upper_err]) for asym error bars."""
    m, lo, hi = triple
    return m, [m - lo, hi - m]


# ----------------------------------------------------------------------------
# Energy-tag control (dead zones)
# ----------------------------------------------------------------------------
ENERGY_TAG = {
    "z=1.00": {"bandpass_E": 2.10, "sup_3layer": 2.54},
    "z=1.04": {"bandpass_E": 2.22, "sup_3layer": 14.71},
    "full_zII_ceiling": 31.0,
}

# ----------------------------------------------------------------------------
# Mechanism / depletion (Fig 5)
# ----------------------------------------------------------------------------
OFFCORE_FRAC = {"signal_z033": 0.121, "photon": 0.014}
ASYM_CORR_SEEDS = [-0.735, -0.638, -0.706]       # signal, seeds 001/002/003
ASYM_CORR_PHOTON = [0.012, 0.023, 0.047]         # photon control
SIGN_MATCH_RATES = [0.070, 0.131, 0.083]         # chance = 0.5
REGION_MEAN_SG = {                               # per-region mean s_G
    "core": {"z=0.33": 49.9, "z=1.00": 20.2},
    "mid": {"z=0.33": 0.0, "z=1.00": 0.0},       # near-null (reported qualitatively)
    "outer": {"z=0.33": 0.0, "z=1.00": 0.0},
    "layer 2": {"z=0.33": 0.0, "z=1.00": 0.0},
}
# Localization argmax = layer-1 core (Table 2, two seeds)
LOCALIZATION_ARGMAX = {
    "z=0.33": (92, 97),
    "z=1.16": (99, 99),
    "z=1.24": (94, 97),
}

# ----------------------------------------------------------------------------
# Collinear ceiling (Fig 6)
# ----------------------------------------------------------------------------
COLLINEAR = {
    "ratio_z033": 0.0028,            # Dx_kin/Dx_resolve at z=0.33, 5 MeV
    "ratio_5MeV_floor": 1e-3,        # -> ~1e-3 collapse at 5 MeV
    "supervised_auc": 0.98,
    "bandpass_auc": 0.991,
    "sigma_miscalib_0p1pct": 7.9,    # 0.1% background-null miscalibration
    "sigma_signal_1pct": 1.6,        # genuine 1% signal admixture
    "bkg_known_to": 0.0002,          # ~0.02%
}

# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
GEOMETRY = {
    "layer_grids": {"L0": (3, 96), "L1": (12, 12), "L2": (12, 6)},
    "n_voxels": 504,
    "core_voxels": 16,
    "two_m_over_E": {"5 GeV": 0.2, "5 MeV": 2e-4},
    "dx_resolve_cm": 7.74,
    "r68_cm": 3.87,
    "masses_GeV": [5, 0.5, 0.05, 0.005],
    "z_m": [0.33, 0.66, 1.00, 1.04, 1.08, 1.16, 1.24],
    "E_chi_GeV": 50,
}

# ----------------------------------------------------------------------------
# Occupancy (Fig S1)
# ----------------------------------------------------------------------------
# Table 3 ladder (alpha=1e-3, seed 001): rungs in order
OCC_LADDER_RUNGS = ["[O0=0]", "E0/Edep", "3-var LR", "Tocc", "pcomb"]
OCC_LADDER = {
    "z=0.33": [0.00, 0.01, 0.00, 0.02, 27.56],
    "z=1.16": [4.59, 20.03, 5.56, 8.27, 11.49],
    "z=1.24": [21.92, 30.35, 21.32, 39.41, 26.54],
}
E0FRAC_BACKFACE = {"z=1.04": 0.026, "z=1.08": 0.001}   # median E0/Edep collapse

# ----------------------------------------------------------------------------
# Preprocessing diagnostic (Fig S2)
# ----------------------------------------------------------------------------
ZI_NORM_OFFSHELL = [159, 108, 262]    # released path, per seed (means 159.4/107.8/262.3)
# Training-consistent (bounded-logit) transform, per seed, under the ANALYSIS
# conditioning (reconstructed E_inc = E_dep/0.937). 3-seed cluster recompute
# flow_I_reparam_e10_allseeds (2026-06-10): seed means 2.081/2.095/2.038.
# NOTE: the older 3.69 was the energy-field conditioning convention
# (flow_I_preproc_check), which the analysis does NOT use -> superseded.
ZI_NORM_ONSHELL_PER_SEED = [2.08, 2.09, 2.04]
ZI_NORM_ONSHELL = 2.08                # seed 001 (analysis-consistent)
D_I = 3
