"""Data layer for the NEEDS-DATA figure panels.

================================================================================
EXPECTED INPUT FILES  (place under data/in/)
================================================================================
calib_pvalues.csv        Fig 3a   col: p                       (~30k held-out photon
                                                                conformal p-values of M)
achieved_vs_nominal.csv  Fig 3b   cols: nominal, achieved       (background FPR sweep)
region_chi2.csv          Fig 3c   cols: region, measured, expected   (OPTIONAL; layer-1
                                                                sub-regions; the 3 layers
                                                                are REAL-NOW from constants)
grid_sic.csv             Fig 4a   cols: mass, z, score, sic     (score in
                                                                {T_raw,M,supervised,global};
                                                                28 cells x scores)
sic_curves.csv           Fig 4c   cols: z, score, rejection, sic, sic_lo, sic_hi
layer1_images.npz        Fig 5a   arrays: photon_mean (12x12), signal_mean (12x12),
                                          [second_photon_xy (2,)]
second_photon.csv        Fig 5a   cols: x, y                    (OPTIONAL fallback if the
                                                                npz lacks second_photon_xy)
asymmetry.csv            Fig 5b   cols: raw_asym, latent_asym, kind  (kind in {signal,photon})
region_mean_sG.csv       Fig 5c   cols: region, mean_sG         (OPTIONAL; all 8 regions;
                                                                else REAL-NOW core values)
collinear_sic.csv        Fig 6a   cols: ratio, sic, [supervised_auc]
e0frac_vs_z.csv          Fig S1a  cols: z, e0_frac_median
event_signal.npz         Fig 2c, 5  array: voxels (504); optional second_photon_xy, argmax_region
event_photon.npz         Fig 2c     array: voxels (504)
score_dist.npz           Fig 5a     arrays: Traw_photon, Traw_signal, M_photon, M_signal

DATA POLICY
-----------
* Real mode (default): load the file; if absent, raise FileNotFoundError with a clear
  message. Never silently fall back to demo.
* Demo mode (demo=True): synthesize stand-in distributions consistent with the reported
  summary statistics, for LAYOUT TESTING ONLY. Figures that synthesize demo data carry a
  conspicuous watermark. Demo synthesis is deterministic (fixed seed).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from style import DATA_IN
import constants as C

_SEED = 0


def _require(name: str) -> Path:
    p = DATA_IN / name
    if not p.exists():
        raise FileNotFoundError(
            f"Required input '{name}' not found at {p}.\n"
            f"  -> Provide it (see data_io.py schema header) or run with --demo for a "
            f"watermarked layout-test render."
        )
    return p


# ----------------------------------------------------------------------------
# Fig 3a  conformal p-values of M
# ----------------------------------------------------------------------------
def load_calib_pvalues(demo: bool = False) -> np.ndarray:
    if demo:
        rng = np.random.default_rng(_SEED)
        return rng.uniform(0.0, 1.0, size=30000)
    return pd.read_csv(_require("calib_pvalues.csv"))["p"].to_numpy()


# ----------------------------------------------------------------------------
# Fig 3b  achieved vs nominal background FPR
# ----------------------------------------------------------------------------
def load_achieved_vs_nominal(demo: bool = False):
    if demo:
        nominal = np.logspace(-3.3, 0.0, 40)
        # diagonal with a soft achievability floor near the per-bin floor
        achieved = np.clip(nominal, C.PER_BIN_FLOOR * 0.9, None)
        return nominal, achieved
    df = pd.read_csv(_require("achieved_vs_nominal.csv"))
    return df["nominal"].to_numpy(), df["achieved"].to_numpy()


# ----------------------------------------------------------------------------
# Fig 3c  OPTIONAL layer-1 sub-region chi^2 (the 3 layers are REAL-NOW)
# ----------------------------------------------------------------------------
def load_region_chi2(demo: bool = False):
    """Return dict {region: (measured, expected)} for sub-regions, or None."""
    if demo:
        # plausible near-diagonal sub-regions for layout
        return {"L1 core": (16.1, 16), "L1 mid": (47.6, 48),
                "L1 outer": (80.3, 80), "L1 left": (71.6, 72), "L1 right": (72.4, 72)}
    p = DATA_IN / "region_chi2.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return {r.region: (r.measured, r.expected) for r in df.itertuples()}


# ----------------------------------------------------------------------------
# Fig 4a  full 28-cell SIC grid
# ----------------------------------------------------------------------------
def load_grid_sic(demo: bool = False) -> pd.DataFrame:
    if demo:
        rng = np.random.default_rng(_SEED)
        masses = C.GEOMETRY["masses_GeV"]
        zs = C.GEOMETRY["z_m"]
        rows = []
        known = {  # exact 5-GeV cells from constants
            ("T_raw", 1.00): C.DEADZONE["z=1.00"]["T_raw"][0],
            ("T_raw", 1.04): C.DEADZONE["z=1.04"]["T_raw"][0],
            ("T_raw", 0.33): C.DEADZONE["z=0.33"]["T_raw"][0],
            ("T_raw", 1.16): C.DEADZONE["z=1.16"]["T_raw"][0],
            ("M", 1.00): C.DEADZONE["z=1.00"]["M"][0],
            ("M", 1.04): C.DEADZONE["z=1.04"]["M"][0],
            ("M", 0.33): C.DEADZONE["z=0.33"]["M"][0],
            ("M", 1.16): C.DEADZONE["z=1.16"]["M"][0],
        }
        for score in ["T_raw", "M"]:
            for m in masses:
                for z in zs:
                    if score == "M" and abs(z - 1.24) < 1e-9:
                        val = np.nan                       # missing -> hatched
                    elif (score, z) in known and abs(m - 5) < 1e-9:
                        val = known[(score, z)]
                    else:
                        base = known.get((score, 1.16), 10.0)
                        # decay with mass and away from the active band
                        val = max(0.0, base * (0.2 + 0.8 * m / 5) * rng.uniform(0.3, 1.0))
                        if score == "T_raw" and z in (1.00, 1.04):
                            val = rng.uniform(0.0, 2.0)
                    rows.append({"mass": m, "z": z, "score": score, "sic": val})
        return pd.DataFrame(rows)
    return pd.read_csv(_require("grid_sic.csv"))


# ----------------------------------------------------------------------------
# Fig 4c  SIC vs background rejection
# ----------------------------------------------------------------------------
def load_sic_curves(demo: bool = False) -> pd.DataFrame:
    if demo:
        rej = np.logspace(1, 4, 30)
        rows = []
        targets = {  # (z, score): plateau SIC
            (1.00, "T_raw"): 0.1, (1.00, "global"): 12.1, (1.00, "M"): 28.65,
            (1.00, "supervised"): 30.96,
            (1.04, "T_raw"): 1.3, (1.04, "global"): 16.2, (1.04, "M"): 30.13,
            (1.04, "supervised"): 31.01,
        }
        for (z, score), plateau in targets.items():
            # smooth rise to plateau vs rejection
            sic = plateau * (1 - np.exp(-rej / 200.0))
            band = 0.06 * plateau
            for r, s in zip(rej, sic):
                rows.append({"z": z, "score": score, "rejection": r,
                             "sic": s, "sic_lo": max(s - band, 0), "sic_hi": s + band})
        return pd.DataFrame(rows)
    return pd.read_csv(_require("sic_curves.csv"))


# ----------------------------------------------------------------------------
# Fig 5a  mean layer-1 images
# ----------------------------------------------------------------------------
def load_layer1_images(demo: bool = False):
    """Return (photon_mean 12x12, signal_mean 12x12, second_photon_xy or None)."""
    if demo:
        rng = np.random.default_rng(_SEED)
        yy, xx = np.mgrid[0:12, 0:12]

        def blob(cx, cy, amp, sig):
            return amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig ** 2))
        photon = blob(5.5, 5.5, 1.0, 1.6) + rng.normal(0, 0.003, (12, 12))
        # signal: core depletion + a faint off-core second photon
        signal = blob(5.5, 5.5, 0.92, 1.6) - blob(5.5, 5.5, 0.10, 1.0) \
            + blob(8.5, 6.5, 0.06, 1.0) + rng.normal(0, 0.003, (12, 12))
        photon = np.clip(photon, 0, None); signal = np.clip(signal, 0, None)
        return photon, signal, np.array([8.5, 6.5])
    p = _require("layer1_images.npz")
    d = np.load(p)
    spx = d["second_photon_xy"] if "second_photon_xy" in d else None
    if spx is None:
        cp = DATA_IN / "second_photon.csv"
        if cp.exists():
            row = pd.read_csv(cp).iloc[0]
            spx = np.array([row["x"], row["y"]])
    return d["photon_mean"], d["signal_mean"], spx


# ----------------------------------------------------------------------------
# Fig 5b  raw vs latent L/R asymmetry
# ----------------------------------------------------------------------------
def load_asymmetry(demo: bool = False) -> pd.DataFrame:
    if demo:
        rng = np.random.default_rng(_SEED)
        n = 6000

        def corr_pair(corr, n, yscale):
            x = rng.uniform(-1, 1, n)
            y = corr * (x / np.std(x)) * yscale + rng.normal(0, yscale, n)
            return x, y
        xs, ys = corr_pair(C.ASYM_CORR_SEEDS[0], n, 12.0)
        xp, yp = corr_pair(C.ASYM_CORR_PHOTON[0], n, 4.0)
        df = pd.DataFrame({
            "raw_asym": np.concatenate([xs, xp]),
            "latent_asym": np.concatenate([ys, yp]),
            "kind": ["signal"] * n + ["photon"] * n,
        })
        return df
    return pd.read_csv(_require("asymmetry.csv"))


# ----------------------------------------------------------------------------
# Fig 5c  OPTIONAL per-region mean s_G (all 8 regions)
# ----------------------------------------------------------------------------
def load_region_mean_sG(demo: bool = False):
    """Return dict {region: mean_sG} for z=0.33, or None to use REAL-NOW core values."""
    if demo:
        return {"L0": 8.0, "L1": 32.0, "L2": 2.8, "L1 core": 49.9, "L1 mid": 0.4,
                "L1 outer": 0.9, "L1 left": 25.0, "L1 right": 20.3}
    p = DATA_IN / "region_mean_sG.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return {r.region: r.mean_sG for r in df.itertuples()}


# ----------------------------------------------------------------------------
# Fig 6a  collinear morphology SIC vs geometric ratio
# ----------------------------------------------------------------------------
def load_collinear_sic(demo: bool = False) -> pd.DataFrame:
    if demo:
        # 4 mass points: ratio falls, SIC collapses below resolvability
        return pd.DataFrame({
            "ratio": [2.843, 0.284, 0.028, 0.0028],
            "sic": [30.05, 0.02, 0.01, 0.006],
            "supervised_auc": [np.nan, np.nan, np.nan, C.COLLINEAR["supervised_auc"]],
        })
    return pd.read_csv(_require("collinear_sic.csv"))


# ----------------------------------------------------------------------------
# Fig S1a  median layer-0 fraction vs z
# ----------------------------------------------------------------------------
def load_e0frac_vs_z(demo: bool = False) -> pd.DataFrame:
    if demo:
        zs = C.GEOMETRY["z_m"]
        # smooth high-front then collapse across the layer-0 back face (1.04 -> 1.08)
        vals = {0.33: 0.030, 0.66: 0.029, 1.00: 0.028, 1.04: 0.026,
                1.08: 0.001, 1.16: 0.0006, 1.24: 0.0004}
        return pd.DataFrame({"z": zs, "e0_frac_median": [vals[z] for z in zs]})
    return pd.read_csv(_require("e0frac_vs_z.csv"))


# ----------------------------------------------------------------------------
# Fig 2c / Fig 5  single-event displays (504-voxel showers)
# ----------------------------------------------------------------------------
def load_event(kind: str, demo: bool = False) -> dict:
    """Shower dict {'voxels'(504) [, 'second_photon_xy', 'argmax_region']} for
    kind in {'signal','photon'}. Real: data/in/event_<kind>.npz. Demo: synthesized
    (two-blob signal / centered photon)."""
    if not demo:
        d = np.load(_require(f"event_{kind}.npz"), allow_pickle=True)
        out = {"voxels": np.asarray(d["voxels"]).reshape(-1)}
        if "second_photon_xy" in d.files:
            out["second_photon_xy"] = np.asarray(d["second_photon_xy"]).reshape(2)
        if "argmax_region" in d.files:
            out["argmax_region"] = str(d["argmax_region"])
        return out
    rng = np.random.default_rng(1 if kind == "signal" else 2)
    yy, xx = np.mgrid[0:12, 0:12]

    def blob(cx, cy, amp, sig):
        return amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig ** 2))
    if kind == "signal":
        l1 = blob(4.6, 5.5, 0.85, 1.25) + blob(8.2, 6.4, 0.5, 1.0)   # core + 2nd photon
        spx = np.array([8.2, 6.4])
    else:
        l1 = blob(5.5, 5.5, 1.0, 1.6); spx = None
    l1 = np.clip(l1 + rng.normal(0, 0.004, (12, 12)), 0, None)
    y0, x0 = np.mgrid[0:3, 0:96]
    l0 = 0.25 * np.exp(-((x0 - 48) ** 2) / (2 * 9 ** 2) - ((y0 - 1) ** 2) / (2 * 0.8 ** 2))
    y2, x2 = np.mgrid[0:12, 0:6]
    l2 = 0.18 * np.exp(-((x2 - 3) ** 2 + (y2 - 6) ** 2) / (2 * 2.2 ** 2))
    out = {"voxels": np.concatenate([l0.ravel(), l1.ravel(), l2.ravel()]),
           "argmax_region": "L1 core"}
    if spx is not None:
        out["second_photon_xy"] = spx
    return out


def load_score_dist(demo: bool = False) -> dict:
    """Per-event Traw and M for photon-test vs the z=1.00 dead-zone signal (for the
    score-distribution panel). Real: data/in/score_dist.npz. Demo: synthesized."""
    if not demo:
        d = np.load(_require("score_dist.npz"))
        return {k: d[k] for k in d.files}
    rng = np.random.default_rng(0); n = 20000
    return {"Traw_photon": rng.normal(0, 1, n), "Traw_signal": rng.normal(0.5, 1.1, n),
            "M_photon": rng.normal(0, 1, n), "M_signal": rng.normal(9, 1.6, n)}
