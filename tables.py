#!/usr/bin/env python3
"""One function per paper table; each prints LaTeX body rows from committed inputs.

Table -> source generator (paper_build/v2/):
  Table I    <- T1.py        (the factorization residual is recomputed here from
                              the committed logp_{total,I,II}.npy arrays)
  Table III  <- T2.py
  Table IV   <- T_oracle.py
  Table V    <- T_arch.py
  Table grid <- T3.py
"""
from __future__ import annotations
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import constants as C  # noqa: E402


# Table I <- paper_build/v2/T1.py  (background-null diagnostics)
def table1():
    def factorization_residual():
        d = REPO / "data" / "in"            # relocated: logp_* committed here (was eval_gamma2_seed_001/)
        lt = np.load(d / "logp_total.npy")
        lI = np.load(d / "logp_I.npy")
        lII = np.load(d / "logp_II.npy")
        resid = np.abs(lt - (lI + lII))
        return resid.max(), lt.size

    print("=== (i) layer chi^2 nulls (constants.CHI2_LAYERS) ===")
    for name, (meas, dof) in C.CHI2_LAYERS.items():
        lab = name.replace("layer ", "L")
        print(rf"\hspace{{1em}}{lab}               & {meas:.1f} & {dof} \\")

    print("\n=== (ii) L1 sub-regions (data/in/region_chi2.csv) ===")
    df = pd.read_csv(REPO / "data" / "in" / "region_chi2.csv")
    for _, r in df.iterrows():
        reg = r["region"].replace("L1 ", "")
        print(rf"\hspace{{1em}}L1, {reg:<6}        & {r['measured']:.2f} & {int(r['expected'])} \\")

    print("\n=== (iii) core photon s_G (constants.CORE_PHOTON_SG) ===")
    sg = C.CORE_PHOTON_SG
    print(rf"core photon $s_G$ mean / std & ${sg['mean']:.2f}$ / ${sg['std']:.2f}$ & --- \\")

    print("\n=== (iv) ||z_II||^2 per seed (constants.ZII_NORM_PER_SEED, D_II) ===")
    for i, v in enumerate(C.ZII_NORM_PER_SEED, start=1):
        print(rf"\hspace{{1em}}seed {i:03d}            & {v:.2f} & {C.D_II} \\")

    print("\n=== (v) factorization residual (recomputed, CPU) ===")
    rmax, n = factorization_residual()
    print(f"max|logp_total-(logp_I+logp_II)| = {rmax!r} over n={n} showers")
    cell = "0" if rmax == 0.0 else f"{rmax:.2e}"
    print(rf"$\max|\log p_{{\rm tot}}-(\log p_{{\rm I}}+\log p_{{\rm II}})|$ "
          rf"& {cell} (exact, $10^5$ showers) & --- \\")


# Table III <- paper_build/v2/T2.py  (energy-tag control + leakage)
def table3():
    # Block A: energy-only controls vs ceiling (from committed constants)
    ET = C.ENERGY_TAG
    ceiling = ET["full_zII_ceiling"]
    blockA = [
        # (z, bandpass total-E SIC, 3-layer-energy supervised SIC, full ceiling)
        ("1.00", ET["z=1.00"]["bandpass_E"], ET["z=1.00"]["sup_3layer"], ceiling),
        ("1.04", ET["z=1.04"]["bandpass_E"], ET["z=1.04"]["sup_3layer"], ceiling),
    ]

    # Block B: leakage controls (read CSV exactly)
    CSV = REPO / "scan_outputs" / "revision" / "supervised_leakage.csv"
    STEM_Z = {"neg_91_5GeV": "0.33", "neg_8_5GeV": "1.16", "zero_5GeV": "1.24"}
    STEM_ORDER = ["neg_91_5GeV", "neg_8_5GeV", "zero_5GeV"]
    MODE_ORDER = [
        "full", "voxel_only", "occupancy_only",
        "metadata_only", "label_shuffle", "cross_file_split",
    ]
    MODE_LABEL = {
        "full": "Full (504 voxel)",
        "voxel_only": "Voxel only (504)",
        "occupancy_only": "Occupancy only (4)",
        "metadata_only": "Layer energy only (3)",
        "label_shuffle": "Label shuffle",
        "cross_file_split": "Cross-file split (504)",
    }

    table = {}  # (stem, mode) -> (AUC, SIC)
    with open(CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            table[(row["stem"], row["mode"])] = (
                float(row["AUC"]), float(row["SIC_at_1e-3"]),
            )

    print("=== BLOCK A: energy-only controls vs ceiling ===")
    print(f"{'z':>5} | {'bandpass-E SIC':>14} | {'3layerE-sup SIC':>15} | {'ceiling':>7}")
    for z, bp, s3, ce in blockA:
        print(f"{z:>5} | {bp:>14.2f} | {s3:>15.2f} | {ce:>7.1f}")
        print(f"LATEX_A & {z} & {bp:.2f} & {s3:.2f} & $\\sim${ce:.0f} \\\\")

    print("\n=== BLOCK B: leakage controls (AUC / SIC@1e-3 per mode per cell) ===")
    header = "{:>22} | ".format("mode") + " | ".join(
        f"z={STEM_Z[s]}" for s in STEM_ORDER
    )
    print(header)
    for mode in MODE_ORDER:
        cells = []
        latex_cells = []
        for stem in STEM_ORDER:
            auc, sic = table[(stem, mode)]
            cells.append(f"{auc:.4f}/{sic:5.2f}")
            latex_cells.append(f"{auc:.4f} & {sic:.2f}")
        print(f"{MODE_LABEL[mode]:>22} | " + " | ".join(cells))
        print(f"LATEX_B {MODE_LABEL[mode]} & " + " & ".join(latex_cells) + r" \\")


# Table IV <- paper_build/v2/T_oracle.py  (oracle-E_inc robustness)
def table4():
    SRC = REPO / "scan_outputs" / "E_inc_robustness_diff.txt"
    SHOW = [(0.33, 5.0), (1.00, 5.0), (1.16, 5.0), (1.24, 5.0)]
    CHAN_ORDER = ["L", "S", "occ", "comb"]
    CHAN_TEX = {
        "L": r"$T_{\mathrm{I}}$",
        "S": r"$T_{S}$",
        "occ": r"$T_{\mathrm{occ}}$",
        "comb": r"$p_{\mathrm{comb}}$",
    }

    def parse():
        rows = {}
        oracle_bin = None
        n_flips = n_cells = None
        line_re = re.compile(
            r"^\s*([\d.]+)\s+([\d.]+)\s+(\w+)\s+([\d.]+)\s+([\d.]+)\s+([+-][\d.]+)\s*$"
        )
        for line in SRC.read_text().splitlines():
            if line.startswith("Oracle bin"):
                oracle_bin = line.split("=", 1)[1].strip()
            if line.startswith("Dominant-channel flips:"):
                tail = line.split(":", 1)[1]
                n_flips, n_cells = (int(x) for x in re.findall(r"\d+", tail))
            m = line_re.match(line)
            if not m:
                continue
            z, mchi, chan = float(m.group(1)), float(m.group(2)), m.group(3)
            sic_def, sic_ora, dlt = float(m.group(4)), float(m.group(5)), float(m.group(6))
            rows[(z, mchi, chan)] = (sic_def, sic_ora, dlt)
        assert oracle_bin and n_flips is not None and n_cells is not None
        return rows, oracle_bin, n_flips, n_cells

    rows, oracle_bin, n_flips, n_cells = parse()

    # Body rows for the displayed cells.
    body = []
    for (z, mchi) in SHOW:
        first = True
        for chan in CHAN_ORDER:
            d, o, dlt = rows[(z, mchi, chan)]
            zcol = f"{z:.2f}" if first else ""
            body.append(
                f"{zcol} & {CHAN_TEX[chan]} & {d:.2f} & {o:.2f} & {dlt:+.2f} \\\\"
            )
            first = False
        body.append(r"\addlinespace")
    if body and body[-1] == r"\addlinespace":
        body.pop()

    # Worst-case |Delta| over the full 28-cell x 4-channel grid (for caption).
    worst_key = max(rows, key=lambda k: abs(rows[k][2]))
    worst_z, worst_m, worst_chan = worst_key
    worst_dlt = rows[worst_key][2]

    print("% --- auto-generated by tables.py table4 (<- paper_build/v2/T_oracle.py) ---")
    print(f"% oracle bin = {oracle_bin}; flips = {n_flips}/{n_cells}")
    print(
        f"% worst |Delta SIC| over all {n_cells} cells x 4 channels: "
        f"{worst_dlt:+.2f} at z={worst_z:.2f}, m={worst_m:g} GeV, channel {worst_chan}"
    )
    print()
    print("\n".join(body))
    print()
    print(
        f"% CAPTION NUMBERS: flips={n_flips}/{n_cells}; "
        f"worst |Delta|={abs(worst_dlt):.2f} ({worst_chan}, z={worst_z:.2f})"
    )


# Table V <- paper_build/v2/T_arch.py  (architecture / hyperparameter / split)
def table5():
    SPLITS = REPO / "data" / "splits"
    # gamma_1 internal train/val split (cluster/split_gamma1.job.yaml): 100k gamma_1
    # showers permuted (seed 42) -> first 70k train, last 30k val.
    GAMMA1_TRAIN, GAMMA1_VAL = 70_000, 30_000

    def load_splits():
        names = ["c0", "c1", "c2", "t"]
        arrs = {n: np.load(SPLITS / f"{n}_idx_seed42.npy") for n in names}
        allidx = np.concatenate([arrs[n] for n in names])
        assert len(allidx) == 100_000, f"split total {len(allidx)} != 100000"
        assert len(set(allidx.tolist())) == 100_000, "split not disjoint/exhaustive"
        return arrs

    arrs = load_splits()
    n_c0, n_c1, n_c2, n_t = (len(arrs[n]) for n in ["c0", "c1", "c2", "t"])

    edges_C0 = np.load(SPLITS / "Econd_bin_edges_5q_C0.npy")
    n_bins = len(edges_C0) - 1

    print("% --- auto-generated by tables.py table5 (<- paper_build/v2/T_arch.py) ---")
    print(f"% gamma_1: train={GAMMA1_TRAIN} / val={GAMMA1_VAL} (seed 42)")
    print(
        f"% gamma_2 4-way (seed 42): C0={n_c0} (occ-PMF) / C1={n_c1} (per-channel cal) "
        f"/ C2={n_c2} (combined cal) / T={n_t} (held-out test); sum={n_c0+n_c1+n_c2+n_t}"
    )
    print(f"% Mondrian: {n_bins} quantile bins in log10(E_cond), edges from C0 fold")
    print()

    print("% Block B body -- gamma_2 split rows:")
    rows = [
        ("$C_0$ (occupancy-PMF fit)", n_c0),
        ("$C_1$ (per-channel calibration)", n_c1),
        ("$C_2$ (combined-score calibration)", n_c2),
        ("$T$ (held-out background test)", n_t),
    ]
    for label, n in rows:
        print(f"    {label} & {n//1000}{{,}}000 \\\\")
    print(f"    \\textit{{total}} & {(n_c0+n_c1+n_c2+n_t)//1000}{{,}}000 \\\\")
    print()

    print("% Mondrian 5-bin edges in log10(E_cond) (E_cond = E_inc / 10 GeV):")
    edge_str = ",\\ ".join(f"{e:+.3f}" for e in edges_C0)
    print(f"%   edges = [{edge_str}]")
    e_inc_lo = 10 ** edges_C0[0] * 10.0
    e_inc_hi = 10 ** edges_C0[-1] * 10.0
    print(
        f"%   physical E_inc span: {e_inc_lo:.3f} GeV ... {e_inc_hi:.3f} GeV "
        f"(10^edge x 10 GeV)"
    )
    print()
    print("% Bin-edge row (for the caption / a footnote):")
    print(f"%   {edge_str}")


# Table grid <- paper_build/v2/T3.py  (full 28-cell x 4-channel SIC grid)
def table_grid():
    CSV = REPO / "scan_outputs/revision/multiseed/sic_grid_4ch_C0split_multiseed.csv"
    LABEL_MAP = {
        "no_stable_label": "--",
        "p_occ": r"$T_{\rm occ}$",
    }
    CHANNELS = ["raw_LL", "p_L", "p_S", "p_occ", "p_comb"]

    def fmt_cell(mean, std):
        return f"{mean:.2f}({std:.2f})"

    def fmt_mass(m):
        s = f"{m:g}"
        return s

    df = pd.read_csv(CSV)
    df = df.sort_values(["z_m", "m_chi_GeV"]).reset_index(drop=True)
    assert len(df) == 28, f"expected 28 cells, got {len(df)}"

    rows = []
    prev_z = None
    for _, r in df.iterrows():
        z = r["z_m"]
        if prev_z is not None and z != prev_z:
            rows.append(r"\addlinespace[2pt]")
        z_str = f"{z:.2f}" if z != prev_z else ""
        prev_z = z

        cells = [fmt_cell(r[f"{c}_mean"], r[f"{c}_std"]) for c in CHANNELS]
        win = LABEL_MAP.get(r["majority_label"], r["majority_label"])
        agree = f"{r['label_agreement_pct']:.0f}"

        line = (
            f"{z_str} & {fmt_mass(r['m_chi_GeV'])} & "
            + " & ".join(cells)
            + f" & {win} & {agree} \\\\"
        )
        rows.append(line)

    print("\n".join(rows))


# dispatch
TABLES = {
    "I": table1,
    "III": table3,
    "IV": table4,
    "V": table5,
    "grid": table_grid,
}


def main():
    ap = argparse.ArgumentParser(description="Print paper table rows from committed inputs.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="print every table (default)")
    g.add_argument("--table", choices=list(TABLES), help="print a single table")
    args = ap.parse_args()
    if args.table:
        TABLES[args.table]()
    else:
        for name, fn in TABLES.items():
            print(f"\n########## TABLE {name} ##########")
            fn()


if __name__ == "__main__":
    main()
