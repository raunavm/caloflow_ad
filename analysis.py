#!/usr/bin/env python3
"""Tier-1 result-producing analysis steps for the calibrated region-resolved AD paper.

Each step re-runs on a laptop from committed inputs and reproduces a committed
scan_outputs/ table byte-for-byte. The GPU/cluster scans that produced the primary
scores are provenance (not included, available on request).

Step  ->  source script  ->  committed output
  supervised_vs_unsupervised:  scripts/supervised_vs_unsupervised_table.py  -> scan_outputs/supervised_vs_unsupervised.{csv,txt}
  aggregate_across_seeds:  scripts/revision/aggregate_across_seeds.py  -> scan_outputs/revision/multiseed/*_multiseed.csv + aggregate_summary.json
  mechanism_map_multiseed:  scripts/revision/make_mechanism_map_multiseed.py  -> scan_outputs/revision/multiseed/mechanism_map_label_table.csv
  matched_epsB_headline:  scripts/revision/matched_epsB_headline.py  -> scan_outputs/revision/multiseed/delta_sic_headline_*.{csv,txt}
  mle_einc_robustness:  scripts/revision/mle_einc_robustness.py  -> scan_outputs/revision/<seed>/mle_einc_bracketing.csv

  python analysis.py --all                        # run every step
  python analysis.py --step matched_epsB_headline # run one step
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))


# supervised_vs_unsupervised
def supervised_vs_unsupervised():
    ROOT = REPO
    SUP_DIR = ROOT / "scan_outputs" / "supervised"
    SIC_CSV = ROOT / "scan_outputs" / "sic_grid_4ch.csv"
    OUT_DIR = ROOT / "scan_outputs"

    CELLS = [
        ("neg_91_5GeV", 0.33, 5.0, "morphology"),
        ("neg_8_5GeV",  1.16, 5.0, "combination"),
        ("zero_5GeV",   1.24, 5.0, "occupancy"),
    ]

    conf = {}
    for r in csv.DictReader(open(SIC_CSV)):
        conf[(float(r["z_m"]), float(r["m_chi_GeV"]))] = r

    rows = []
    for stem, z, m, regime in CELLS:
        sup = json.load(open(SUP_DIR / f"{stem}.json"))
        c = conf[(z, m)]
        chs = {"T_raw": float(c["raw_LL"]), "T_I": float(c["p_L"]),
               "T_S": float(c["p_S"]), "T_occ": float(c["p_occ"]),
               "p_comb": float(c["p_comb"])}
        best_unsup_name = max(chs, key=chs.get)
        best_unsup_sic = chs[best_unsup_name]
        sup_sic_1e3 = sup["SIC@0.001"]
        ratio = best_unsup_sic / sup_sic_1e3
        rows.append({
            "regime": regime, "z_m": z, "m_chi_GeV": m,
            "SUP_AUC": sup["AUC"],
            "SUP_SIC_at_1e-2": sup["SIC@0.01"],
            "SUP_SIC_at_1e-3": sup["SIC@0.001"],
            "SUP_eps_B_at_1e-2": sup["eps_B@0.01"],
            "SUP_eps_B_at_1e-3": sup["eps_B@0.001"],
            "T_raw_SIC": chs["T_raw"],
            "T_I_SIC": chs["T_I"],
            "T_S_SIC": chs["T_S"],
            "T_occ_SIC": chs["T_occ"],
            "p_comb_SIC": chs["p_comb"],
            "best_unsup_channel": best_unsup_name,
            "best_unsup_SIC": best_unsup_sic,
            "best_unsup_over_supervised": ratio,
            # Theoretical SIC ceiling at α=1e-3 (assuming continuous score, ε_B = α exactly)
            "theory_max_SIC_1e-3": 1.0 / (1e-3) ** 0.5,
        })

    with open(OUT_DIR / "supervised_vs_unsupervised.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    lines = [
        "=" * 130,
        "Figure 6.4 — supervised baseline (upper-bound sanity check) vs unsupervised channels",
        "=" * 130,
        "",
        "Setup: MLP [504, 256, 128, 64, 1] with BatchNorm+ReLU+Dropout(0.2). AdamW lr=1e-3.",
        "Train: 80% (80k bg + 80k sig). Test: 20% (20k bg + 20k sig). seed=42. 15 epochs, A40 GPU.",
        "Theoretical SIC ceiling at α=1e-3 with continuous score and ε_B = α: 1/√α ≈ 31.62.",
        "",
        f"  {'regime':>12s}  {'cell (z, m_χ)':>15s}    "
        f"{'SUP AUC':>8s}  {'SUP SIC@1e-2':>13s}  {'SUP SIC@1e-3':>13s}  "
        f"{'best unsup':>10s}  {'unsup SIC':>9s}  {'unsup/sup':>9s}",
        "-" * 130,
    ]
    for r in rows:
        lines.append(
            f"  {r['regime']:>12s}  z={r['z_m']:.2f} m={r['m_chi_GeV']:g}      "
            f"{r['SUP_AUC']:>8.4f}  {r['SUP_SIC_at_1e-2']:>13.2f}  {r['SUP_SIC_at_1e-3']:>13.2f}  "
            f"{r['best_unsup_channel']:>10s}  {r['best_unsup_SIC']:>9.2f}  "
            f"{r['best_unsup_over_supervised']:>9.2f}"
        )
    lines += [
        "-" * 130,
        "",
        "Per-channel unsupervised SICs at α=1e-3 (for §6.4 narrative):",
        f"  {'regime':>12s}  {'T_raw':>6s}  {'T_I':>6s}  {'T_S':>6s}  {'T_occ':>6s}  {'p_comb':>7s}  "
        f"{'SUP_1e-3':>9s}",
    ]
    for r in rows:
        lines.append(f"  {r['regime']:>12s}  {r['T_raw_SIC']:>6.2f}  {r['T_I_SIC']:>6.2f}  "
                     f"{r['T_S_SIC']:>6.2f}  {r['T_occ_SIC']:>6.2f}  {r['p_comb_SIC']:>7.2f}  "
                     f"{r['SUP_SIC_at_1e-3']:>9.2f}")
    lines += [
        "",
        "Interpretation (per user §6.4 paragraph spec):",
        "  - Supervised AUC ≈ 1 in all 3 cells → 504-dim voxel input contains essentially complete",
        "    discrimination information for these signals.",
        "  - Supervised SIC saturates at the theoretical 1/√α ceiling (≈ 31.62 at α=1e-3) since",
        "    the continuous classifier score gives ε_B = α exactly.",
        "  - Unsupervised performance vs supervised, per regime:",
        "      morphology   (z=0.33):  T_S=30.05 vs sup=31.52 → 95% of supervised, achieves",
        "                              comparable SIC without signal labels",
        "      combination  (z=1.16):  p_comb=13.07 vs sup=31.59 → 41% of supervised; the",
        "                              combination is bounded by Tippett multiple-testing penalty",
        "      occupancy    (z=1.24):  T_occ=40.70 vs sup=31.62 → 129% of supervised; T_occ's",
        "                              discrete achievability floor (ε_B < α) lets it exceed the",
        "                              nominal-α SIC ceiling",
        "",
        "Paper claim: supervised provides strong discrimination when the signal hypothesis is known.",
        "Our method is background-only, requires no signal labels, and provides mechanism attribution",
        "via calibrated per-channel p-values. In the occupancy regime the unsupervised method exceeds",
        "the nominal-α supervised ceiling via discrete-score achievability.",
    ]
    out = "\n".join(lines)
    print(out)
    (OUT_DIR / "supervised_vs_unsupervised.txt").write_text(out + "\n")
    print(f"\n[wrote] {OUT_DIR}/supervised_vs_unsupervised.{{csv,txt}}")


# aggregate_across_seeds
def aggregate_across_seeds():
    import csv
    import json
    import math
    import statistics
    from collections import Counter

    ROOT = REPO
    REV = ROOT / "scan_outputs" / "revision"

    DEFAULT_SEEDS = ["seed_001", "seed_002", "seed_003"]

    # Per-EXP config: (per-seed filename, list of channel columns to aggregate,
    #                  cell-key columns, optional dominant-channel column to vote on)
    EXP_CONFIGS = {
        "sic_grid_4ch_C0split": {
            "channels": ["raw_LL", "T_occ_raw", "T_occ_simple",
                         "p_raw", "p_L", "p_S", "p_occ", "p_Bonf", "p_comb"],
            "cell_keys": ["z_m", "m_chi_GeV", "stem"],
            "dom_candidates": ["p_raw", "p_L", "p_S", "p_occ", "p_comb"],
        },
        "matched_epsB_table": {
            # variable number of target_<eps>_SIC columns; let aggregator discover.
            "channels": None,                    # discover from header
            "channel_prefix": "target_",
            "cell_keys": ["stem", "z_m", "m_chi_GeV", "channel"],
            "dom_candidates": None,
        },
        "random_tiebreak_table": {
            "channels": ["cons_SIC_raw", "cons_SIC_L", "cons_SIC_S",
                         "cons_SIC_occ", "cons_SIC_comb",
                         "rand_SIC_raw", "rand_SIC_L", "rand_SIC_S",
                         "rand_SIC_occ", "rand_SIC_comb"],
            "cell_keys": ["stem", "z_m", "m_chi_GeV"],
            "dom_candidates": None,
        },
        "bincount_scan": {
            # nb3_SIC_raw, nb5_SIC_raw, ... nb10_SIC_comb
            "channels": None,
            "channel_prefix": "nb",
            "cell_keys": ["stem", "z_m", "m_chi_GeV"],
            "dom_candidates": None,
        },
        "sic_grid_evalue": {
            "channels": ["SIC_comb_evalue"],
            "cell_keys": ["stem", "z_m", "m_chi_GeV"],
            "dom_candidates": None,
        },
        "sic_grid_cauchy": {
            "channels": ["SIC_comb_cauchy"],
            "cell_keys": ["stem", "z_m", "m_chi_GeV"],
            "dom_candidates": None,
        },
        "sic_grid_5ch_with_shellII": {
            "channels": ["sic_raw", "sic_L", "sic_S", "sic_occ", "sic_shellII"],
            "cell_keys": ["stem", "z_m", "m_chi_GeV"],
            "dom_candidates": ["sic_raw", "sic_L", "sic_S", "sic_occ", "sic_shellII"],
        },
        "mle_einc_bracketing": {
            "channels": ["A_p_raw", "A_p_L", "A_p_S", "A_p_occ", "A_p_comb",
                         "D_p_raw", "D_p_L", "D_p_S", "D_p_occ", "D_p_comb"],
            "cell_keys": ["stem", "z_m", "m_chi_GeV"],
            "dom_candidates": None,
        },
        "sic_grid_4ch_oracle_symmetric": {
            "channels": ["raw", "L", "S", "occ", "comb"],
            "cell_keys": ["stem", "z_m", "m_chi_GeV"],
            "dom_candidates": ["raw", "L", "S", "occ", "comb"],
        },
    }

    # PC-2: dominance rule. A cell has a dominant channel if top SIC > sqrt(2)*runner_up
    # OR top SIC > DOM_FLOOR and runner_up < 0.5*top.
    SQRT2 = math.sqrt(2.0)
    DOM_FLOOR = 5.0
    NO_LABEL = "no_stable_label"
    NO_DOM   = "no_dominance"

    def fnum(x):
        try:
            return float(x)
        except (ValueError, TypeError):
            return None

    def per_cell_dominant(row, candidates):
        """Return (label, has_dominance) for a single per-seed row."""
        sics = {c: fnum(row.get(c)) for c in candidates}
        sics = {c: v for c, v in sics.items() if v is not None}
        if not sics:
            return (NO_DOM, False)
        ordered = sorted(sics.items(), key=lambda kv: -kv[1])
        top_ch, top_v = ordered[0]
        if len(ordered) >= 2:
            run_v = ordered[1][1]
        else:
            run_v = 0.0
        has_dom = (top_v > DOM_FLOOR) and (top_v > SQRT2 * max(run_v, 1e-12))
        return (top_ch if has_dom else NO_DOM, has_dom)

    def aggregate_cell(per_seed_rows, channels, dom_candidates):
        """Per-cell aggregation: mean/std per channel, plus PC-1/PC-2 metrics."""
        out = {}
        for ch in channels:
            vals = [fnum(r.get(ch)) for r in per_seed_rows]
            vals = [v for v in vals if v is not None]
            if not vals:
                out[f"{ch}_mean"] = None
                out[f"{ch}_std"]  = None
                continue
            out[f"{ch}_mean"] = statistics.mean(vals)
            out[f"{ch}_std"]  = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            out[f"{ch}_n"]    = len(vals)

        # PC-1/PC-2 dominance voting
        if dom_candidates:
            votes = [per_cell_dominant(r, dom_candidates) for r in per_seed_rows]
            labels = [v[0] for v in votes]
            dom_flags = [v[1] for v in votes]
            ctr = Counter(labels)
            # Strip the NO_DOM "label" from majority voting; it isn't a channel.
            ctr_real = Counter({k: v for k, v in ctr.items() if k != NO_DOM})
            if ctr_real:
                most_common, n_top = ctr_real.most_common(1)[0]
            else:
                most_common, n_top = (NO_LABEL, 0)
            # PC-1: only crown a winner if it has >= ceil(N/2) votes; else NO_LABEL.
            n_total = len(per_seed_rows)
            threshold = (n_total + 1) // 2   # majority (e.g. 2 of 3, 3 of 5)
            if n_top < threshold:
                label_winner = NO_LABEL
            else:
                label_winner = most_common
            # PC-2 metrics
            label_agree_pct = 100.0 * n_top / n_total if n_total else 0.0
            dom_agree_pct   = 100.0 * sum(dom_flags) / n_total if n_total else 0.0
            out["majority_label"]   = label_winner
            out["label_agreement_pct"]    = label_agree_pct
            out["dominance_agreement_pct"] = dom_agree_pct
            out["per_seed_labels"]   = ";".join(labels)
        return out

    def aggregate_exp(exp_name, seeds, cfg):
        rows_per_seed = []
        for s in seeds:
            path = REV / s / f"{exp_name}.csv"
            if not path.exists():
                print(f"  [skip] missing {path}")
                return None
            rows_per_seed.append(list(csv.DictReader(open(path))))

        # Discover channels if needed
        channels = cfg.get("channels")
        if channels is None and "channel_prefix" in cfg:
            header = list(rows_per_seed[0][0].keys()) if rows_per_seed[0] else []
            prefix = cfg["channel_prefix"]
            channels = [h for h in header if h.startswith(prefix)]
        if channels is None:
            raise RuntimeError(f"no channels specified for {exp_name}")
        dom_candidates = cfg.get("dom_candidates")
        cell_keys = cfg["cell_keys"]

        # Index by cell key
        indexed = []
        for rows in rows_per_seed:
            idx = {tuple(r[k] for k in cell_keys): r for r in rows}
            indexed.append(idx)
        all_keys = set()
        for idx in indexed:
            all_keys |= set(idx.keys())

        out_rows = []
        for k in sorted(all_keys, key=lambda t: tuple(fnum(x) if fnum(x) is not None else str(x) for x in t)):
            per_seed = [idx[k] for idx in indexed if k in idx]
            if len(per_seed) < len(seeds):
                print(f"  [partial] cell {k} only in {len(per_seed)}/{len(seeds)} seeds")
            agg = aggregate_cell(per_seed, channels, dom_candidates)
            out_row = {ck: v for ck, v in zip(cell_keys, k)}
            out_row["n_seeds"] = len(per_seed)
            out_row.update(agg)
            out_rows.append(out_row)

        return out_rows

    seeds = list(DEFAULT_SEEDS)
    exps  = list(EXP_CONFIGS)

    out_root = REV / "multiseed"
    out_root.mkdir(parents=True, exist_ok=True)

    summary = {}
    for exp in exps:
        cfg = EXP_CONFIGS[exp]
        print(f"\n[aggregate] {exp}  seeds={seeds}")
        rows = aggregate_exp(exp, seeds, cfg)
        if rows is None:
            summary[exp] = "missing per-seed input"
            continue
        out_path = out_root / f"{exp}_multiseed.csv"
        fieldnames = list(rows[0].keys()) if rows else []
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows: w.writerow(r)
        print(f"  -> {out_path}  ({len(rows)} cells)")
        summary[exp] = f"ok ({len(rows)} cells)"

    (out_root / "aggregate_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[summary]\n" + json.dumps(summary, indent=2))


# mechanism_map_multiseed
def mechanism_map_multiseed():
    from pathlib import Path
    import csv
    import json
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    ROOT = REPO
    SPLITS = ROOT / "data" / "splits"
    MULTI_CSV = ROOT / "scan_outputs" / "revision" / "multiseed" / "sic_grid_4ch_C0split_multiseed.csv"
    FIG_DIR = ROOT / "figures" / "revision"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR = ROOT / "scan_outputs" / "revision" / "multiseed"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    M_GRID = np.array([0.005, 0.05, 0.5, 5.0])
    Z_GRID = np.array([0.33, 0.66, 1.00, 1.04, 1.08, 1.16, 1.24])

    # Category encoding (Panel A)
    CAT = {
        "no_detection":         0,
        "no_dominance_codet":   1,   # multi-channel co-detection, no single channel dominates
        "no_stable_label":      2,
        "p_L":                  3,
        "p_S":                  4,
        "p_occ":                5,
        "p_comb":               6,
    }
    CAT_COLORS = [
        "#ffffff",   # no_detection — white (truly nothing above SIC floor)
        "#e6e6fa",   # no_dominance_codet — pale lavender (co-detected, no dominance)
        "#999999",   # no_stable_label — darker grey (PC-1: seeds disagree)
        "#d62728",   # p_L — red
        "#1f77b4",   # p_S — blue
        "#2ca02c",   # p_occ — green
        "#ff7f0e",   # p_comb — orange
    ]
    CAT_LABELS = [
        "no detection (all channels SIC $<$ 5 in every seed)",
        "no dominance (multi-channel co-detection across all 3 seeds)",
        r"no stable label (seeds disagree on dominant channel)",
        "$p_L$ dominant (longitudinal aggregate)",
        "$p_S$ dominant (morphology)",
        "$p_{\\rm occ}$ dominant (decay-position aggregate)",
        "$p_{\\rm comb}$ dominant (nested-conformal combination)",
    ]

    # SIC floor below which a cell is "no_detection"
    SIC_FLOOR = 5.0

    def grid_lookup(rows, z, m):
        """Return the aggregated row for (z, m), or None."""
        for r in rows:
            if abs(float(r["z_m"]) - z) < 1e-6 and abs(float(r["m_chi_GeV"]) - m) < 1e-6:
                return r
        return None

    def category_for(row):
        if row is None:
            return CAT["no_detection"]
        dom_agree = float(row.get("dominance_agreement_pct", 0.0) or 0.0)
        label_agree = float(row.get("label_agreement_pct", 0.0) or 0.0)
        label = (row.get("majority_label") or "").strip()

        # Look up the maximum per-channel mean SIC across the four primary channels.
        # If every channel has mean SIC < SIC_FLOOR in this cell, it's true no-detection.
        max_sic = 0.0
        for col in ("p_raw_mean", "p_L_mean", "p_S_mean", "p_occ_mean", "p_comb_mean"):
            try:
                v = float(row.get(col, 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if v > max_sic:
                max_sic = v

        if max_sic < SIC_FLOOR:
            return CAT["no_detection"]

        # Cell IS detected. Now distinguish:
        #   - "no_dominance_codet": all 3 seeds agree no single channel dominates
        #   - "no_stable_label":    seeds DISAGREE about which channel dominates
        #   - "<channel>":          ≥2/3 seeds agree on a specific channel
        if dom_agree == 0.0:
            return CAT["no_dominance_codet"]
        if label == "no_stable_label" or label == "":
            return CAT["no_stable_label"]
        if label not in CAT:
            return CAT["no_stable_label"]
        return CAT[label]

    def build_grids(rows):
        cat = np.full((len(M_GRID), len(Z_GRID)), CAT["no_detection"], dtype=int)
        label_agree = np.full(cat.shape, np.nan)
        dom_agree   = np.full(cat.shape, np.nan)
        per_seed_labels = np.full(cat.shape, "", dtype=object)
        for i, m in enumerate(M_GRID):
            for j, z in enumerate(Z_GRID):
                r = grid_lookup(rows, z, m)
                cat[i, j] = category_for(r)
                if r is not None:
                    label_agree[i, j] = float(r.get("label_agreement_pct") or 0.0)
                    dom_agree[i, j]   = float(r.get("dominance_agreement_pct") or 0.0)
                    per_seed_labels[i, j] = r.get("per_seed_labels", "")
        return cat, label_agree, dom_agree, per_seed_labels

    def render():
        if not MULTI_CSV.exists():
            print(f"[abort] multi-seed CSV missing: {MULTI_CSV}")
            print(f"        run aggregate_across_seeds.py first.")
            return
        rows = list(csv.DictReader(open(MULTI_CSV)))
        cat, lab_a, dom_a, per_seed = build_grids(rows)

        # 3-row figure: Panel A (categorical), Panel B-1 (label %), Panel B-2 (dom %)
        fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True,
                                  gridspec_kw={"height_ratios": [1.2, 1.0, 1.0]})

        # Panel A: categorical heatmap
        cmap_A = ListedColormap(CAT_COLORS)
        norm_A = BoundaryNorm(np.arange(-0.5, len(CAT) + 0.5, 1.0), cmap_A.N)
        im_A = axes[0].imshow(cat, aspect="auto", cmap=cmap_A, norm=norm_A,
                               origin="lower")
        axes[0].set_yticks(np.arange(len(M_GRID)))
        axes[0].set_yticklabels([f"{m:g}" for m in M_GRID])
        axes[0].set_ylabel(r"$m_\chi$ (GeV)")
        axes[0].set_title("Panel A — Majority dominant channel across seeds  (PC-1)")
        for i in range(len(M_GRID)):
            for j in range(len(Z_GRID)):
                txt = per_seed[i, j] if per_seed[i, j] else "—"
                txt_short = txt.replace("no_dominance", "·").replace("p_", "")
                axes[0].text(j, i, txt_short, ha="center", va="center", fontsize=6.5,
                              color="black", wrap=True)
        # Legend
        legend_handles = [Patch(facecolor=CAT_COLORS[i], edgecolor="black",
                                 label=CAT_LABELS[i]) for i in range(len(CAT))]
        axes[0].legend(handles=legend_handles, loc="upper center",
                        bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=7,
                        frameon=False)

        # Panel B-1: label agreement %
        im_B1 = axes[1].imshow(lab_a, aspect="auto", cmap="RdYlGn",
                                vmin=0, vmax=100, origin="lower")
        axes[1].set_yticks(np.arange(len(M_GRID)))
        axes[1].set_yticklabels([f"{m:g}" for m in M_GRID])
        axes[1].set_ylabel(r"$m_\chi$ (GeV)")
        axes[1].set_title("Panel B1 — Label agreement %  (PC-2)")
        for i in range(len(M_GRID)):
            for j in range(len(Z_GRID)):
                if not np.isnan(lab_a[i, j]):
                    axes[1].text(j, i, f"{lab_a[i,j]:.0f}", ha="center", va="center",
                                  fontsize=8,
                                  color="white" if lab_a[i,j] < 50 else "black")
        cb1 = fig.colorbar(im_B1, ax=axes[1], shrink=0.8)
        cb1.set_label("% seeds agree on plurality channel")

        # Panel B-2: dominance agreement %
        im_B2 = axes[2].imshow(dom_a, aspect="auto", cmap="RdYlGn",
                                vmin=0, vmax=100, origin="lower")
        axes[2].set_xticks(np.arange(len(Z_GRID)))
        axes[2].set_xticklabels([f"{z:.2f}" for z in Z_GRID])
        axes[2].set_yticks(np.arange(len(M_GRID)))
        axes[2].set_yticklabels([f"{m:g}" for m in M_GRID])
        axes[2].set_xlabel("displacement $z$ (m)")
        axes[2].set_ylabel(r"$m_\chi$ (GeV)")
        axes[2].set_title("Panel B2 — Dominance agreement %  (PC-2)")
        for i in range(len(M_GRID)):
            for j in range(len(Z_GRID)):
                if not np.isnan(dom_a[i, j]):
                    axes[2].text(j, i, f"{dom_a[i,j]:.0f}", ha="center", va="center",
                                  fontsize=8,
                                  color="white" if dom_a[i,j] < 50 else "black")
        cb2 = fig.colorbar(im_B2, ax=axes[2], shrink=0.8)
        cb2.set_label("% seeds with ANY dominant channel")

        plt.tight_layout()
        for ext in ("pdf", "png"):
            path = FIG_DIR / f"mechanism_map_multiseed.{ext}"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            print(f"[saved] {path}")
        plt.close(fig)

        # Companion table
        table_rows = []
        for i, m in enumerate(M_GRID):
            for j, z in enumerate(Z_GRID):
                r = grid_lookup(rows, z, m)
                if r is None: continue
                cat_idx = cat[i, j]
                label_name = [k for k, v in CAT.items() if v == cat_idx][0]
                table_rows.append({
                    "z_m": z, "m_chi_GeV": m,
                    "majority_channel": label_name,
                    "label_agreement_pct": float(r.get("label_agreement_pct", 0) or 0),
                    "dominance_agreement_pct": float(r.get("dominance_agreement_pct", 0) or 0),
                    "per_seed_labels": r.get("per_seed_labels", ""),
                    "n_seeds": int(r.get("n_seeds", 0)),
                })
        out_csv = OUT_DIR / "mechanism_map_label_table.csv"
        fieldnames = list(table_rows[0].keys())
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in table_rows: w.writerow(r)
        print(f"[saved] {out_csv}")

    render()


# matched_epsB_headline
def matched_epsB_headline():
    """E1 — capability delta (ΔSIC) of the decomposed/combined channels vs the Krause scalar baseline."""
    from pathlib import Path
    import csv

    ROOT = REPO
    MULTI = ROOT / "scan_outputs" / "revision" / "multiseed"
    GRID = MULTI / "sic_grid_4ch_C0split_multiseed.csv"
    MATCHED = MULTI / "matched_epsB_table_multiseed.csv"

    def fnum(x, default=0.0):
        try:
            return float(x)
        except (TypeError, ValueError):
            return default

    # View A: 28-cell ΔSIC at nominal α = 1e-3
    rows_28 = []
    with open(GRID) as f:
        for r in csv.DictReader(f):
            p_raw = fnum(r["p_raw_mean"])
            p_L = fnum(r["p_L_mean"]); p_S = fnum(r["p_S_mean"]); p_occ = fnum(r["p_occ_mean"])
            p_comb = fnum(r["p_comb_mean"])
            best = max(p_L, p_S, p_occ)
            which = max((p_L, "p_L"), (p_S, "p_S"), (p_occ, "p_occ"))[1]
            rows_28.append({
                "z_m": r["z_m"], "m_chi_GeV": r["m_chi_GeV"], "stem": r["stem"],
                "p_raw": f"{p_raw:.3f}",
                "best_channel": which, "best_SIC": f"{best:.3f}",
                "p_comb": f"{p_comb:.3f}",
                "dSIC_best": f"{best - p_raw:+.3f}",
                "dSIC_comb": f"{p_comb - p_raw:+.3f}",
            })

    with open(MULTI / "delta_sic_headline_28cell.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_28[0].keys()))
        w.writeheader()
        for r in rows_28:
            w.writerow(r)

    # View B: matched-ε_B ΔSIC for the headline cells. For each (stem, z, m), pick the
    # occupancy channel's achieved ε_B as the matched operating point and compare raw vs
    # best vs comb at that same ε_B. The matched table is keyed by (stem, channel) and
    # tabulates SIC at a set of ε_B targets.
    matched = {}
    with open(MATCHED) as f:
        for r in csv.DictReader(f):
            key = (r["stem"], r["z_m"], r["m_chi_GeV"])
            matched.setdefault(key, {})[r["channel"]] = r

    TARGETS = ["1e-02", "2e-03", "1e-03", "5e-04", "2e-04", "1e-04"]
    rows_m = []
    for key, chans in matched.items():
        stem, z, m = key
        raw = chans.get("raw", {})
        comb = chans.get("comb", {})
        occ = chans.get("occ", {})
        L = chans.get("L", {}); S = chans.get("S", {})
        sup = chans.get("SUP", {})
        for tgt in TARGETS:
            col = f"target_{tgt}_SIC_mean"
            eb_col = f"target_{tgt}_eb_mean"
            raw_sic = fnum(raw.get(col)); comb_sic = fnum(comb.get(col))
            occ_sic = fnum(occ.get(col)); L_sic = fnum(L.get(col)); S_sic = fnum(S.get(col))
            sup_sic = fnum(sup.get(col)) if sup.get(col) not in (None, "", "nan") else None
            best = max(L_sic, S_sic, occ_sic)
            rows_m.append({
                "stem": stem, "z_m": z, "m_chi_GeV": m, "epsB_target": tgt,
                "epsB_achieved_raw": raw.get(eb_col, ""),
                "raw_SIC": f"{raw_sic:.3f}",
                "best_SIC": f"{best:.3f}",
                "comb_SIC": f"{comb_sic:.3f}",
                "occ_SIC": f"{occ_sic:.3f}",
                "sup_SIC": ("" if sup_sic is None else f"{sup_sic:.3f}"),
                "dSIC_best": f"{best - raw_sic:+.3f}",
                "dSIC_comb": f"{comb_sic - raw_sic:+.3f}",
            })

    with open(MULTI / "delta_sic_headline_matched.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_m[0].keys()))
        w.writeheader()
        for r in rows_m:
            w.writerow(r)

    # Human-readable summary
    lines = ["E1 — capability delta vs Krause scalar baseline (p_raw)", "=" * 70, ""]
    lines.append("View A: 28-cell ΔSIC at nominal α = 1e-3 (cells with any detection)")
    lines.append(f"{'z':>5s} {'m(GeV)':>7s} {'p_raw':>7s} {'best':>7s} {'(ch)':>6s} "
                 f"{'comb':>7s} {'Δbest':>7s} {'Δcomb':>7s}")
    for r in sorted(rows_28, key=lambda x: (float(x["m_chi_GeV"]), float(x["z_m"]))):
        if max(fnum(r["best_SIC"]), fnum(r["p_comb"]), fnum(r["p_raw"])) < 5.0:
            continue
        lines.append(f"{float(r['z_m']):5.2f} {float(r['m_chi_GeV']):7.3f} "
                     f"{fnum(r['p_raw']):7.2f} {fnum(r['best_SIC']):7.2f} "
                     f"{r['best_channel']:>6s} {fnum(r['p_comb']):7.2f} "
                     f"{fnum(r['dSIC_best']):+7.2f} {fnum(r['dSIC_comb']):+7.2f}")

    n_best_pos = sum(1 for r in rows_28 if fnum(r["dSIC_best"]) > 0)
    n_comb_pos = sum(1 for r in rows_28 if fnum(r["dSIC_comb"]) > 0)
    worst_comb = min(fnum(r["dSIC_comb"]) for r in rows_28)
    worst_best = min(fnum(r["dSIC_best"]) for r in rows_28)
    lines.append("")
    lines.append(f"Across 28 cells: Δbest > 0 in {n_best_pos}/28; Δcomb > 0 in {n_comb_pos}/28.")
    lines.append(f"Worst-case Δbest = {worst_best:+.2f} SIC; worst-case Δcomb = {worst_comb:+.2f} SIC.")
    lines.append("")
    lines.append("View B: matched achieved ε_B, headline cells")
    lines.append(f"{'stem':22s} {'epsB':>6s} {'raw':>7s} {'best':>7s} {'comb':>7s} "
                 f"{'occ':>7s} {'sup':>7s} {'Δbest':>7s} {'Δcomb':>7s}")
    for r in rows_m:
        if r["epsB_target"] not in ("1e-03", "1e-04"):
            continue
        lines.append(f"{r['stem']:22s} {r['epsB_target']:>6s} {fnum(r['raw_SIC']):7.2f} "
                     f"{fnum(r['best_SIC']):7.2f} {fnum(r['comb_SIC']):7.2f} "
                     f"{fnum(r['occ_SIC']):7.2f} "
                     f"{(fnum(r['sup_SIC']) if r['sup_SIC'] else float('nan')):7.2f} "
                     f"{fnum(r['dSIC_best']):+7.2f} {fnum(r['dSIC_comb']):+7.2f}")

    txt = "\n".join(lines) + "\n"
    (MULTI / "delta_sic_headline.txt").write_text(txt)
    print(txt)
    print(f"[wrote] {MULTI}/delta_sic_headline_28cell.csv")
    print(f"[wrote] {MULTI}/delta_sic_headline_matched.csv")
    print(f"[wrote] {MULTI}/delta_sic_headline.txt")


# mle_einc_robustness
def mle_einc_robustness():
    """EXP-14: E_inc reconstruction robustness, bracketed between default (A) and oracle (D) endpoints."""
    from pathlib import Path
    import csv
    import json
    import os

    ROOT = REPO
    SEED_DIR = os.environ.get("SEED_DIR", "seed_001")
    OUT = ROOT / "scan_outputs" / "revision" / SEED_DIR
    OUT.mkdir(parents=True, exist_ok=True)

    # Estimator A: per-seed default conformal SIC (output of signal_conformal_sic_C0split.py).
    # Estimator D: oracle bin (currently a single-seed file produced by the legacy
    # E_inc_robustness.py; per-seed regeneration is queued as a follow-up.
    # When the per-seed oracle CSV exists, prefer it.)
    A_csv_per_seed = ROOT / "scan_outputs" / "revision" / SEED_DIR / "sic_grid_4ch_C0split.csv"
    A_csv_legacy   = ROOT / "scan_outputs" / "revision" / "sic_grid_4ch_C0split.csv"
    A_csv = A_csv_per_seed if A_csv_per_seed.exists() else A_csv_legacy

    D_csv_per_seed = ROOT / "scan_outputs" / "revision" / SEED_DIR / "sic_grid_4ch_oracle.csv"
    D_csv_legacy   = ROOT / "scan_outputs" / "sic_grid_4ch_oracle.csv"
    D_csv = D_csv_per_seed if D_csv_per_seed.exists() else D_csv_legacy

    print(f"[setup] SEED_DIR={SEED_DIR}")
    print(f"        A (default) CSV: {A_csv}")
    print(f"        D (oracle)  CSV: {D_csv}")
    print(f"        OUT:             {OUT}")

    a_rows = list(csv.DictReader(open(A_csv)))
    d_rows = list(csv.DictReader(open(D_csv)))

    # Both files have columns: z_m,m_chi_GeV,stem,raw_LL,T_occ_raw,T_occ_simple,p_raw,p_L,p_S,p_occ,p_Bonf,p_comb
    CHS = ("p_raw","p_L","p_S","p_occ","p_comb")
    def dominant(row):
        return max(CHS, key=lambda c: float(row[c]))

    key = lambda r: (r["stem"], r["z_m"], r["m_chi_GeV"])
    A = {key(r): r for r in a_rows}
    D = {key(r): r for r in d_rows}

    shared = sorted(set(A) & set(D), key=lambda k: (float(k[1]), float(k[2])))
    out_rows = []
    flips = []
    sig_flips = []
    print(f"{'='*100}")
    print(f"EXP-14: bracketing analysis  default (A) vs oracle (D)")
    print(f"{'='*100}")
    print(f"  {'cell':>30s}  {'A dom':>7s}  {'D dom':>7s}  "
          f"{'A p_S':>6s} {'D p_S':>6s}  {'A p_occ':>7s} {'D p_occ':>7s}  flip?")

    for k in shared:
        a = A[k]; d = D[k]
        da = dominant(a); dd = dominant(d)
        flipped = da != dd
        # signature flip if either side > 1.0 in SIC
        sig = flipped and max(float(a[da]), float(d[dd])) > 1.0
        if flipped: flips.append(k)
        if sig: sig_flips.append((k, da, dd, a, d))
        print(f"  {k[0][-26:]:>30s}  {da[2:]:>7s}  {dd[2:]:>7s}  "
              f"{float(a['p_S']):>6.2f} {float(d['p_S']):>6.2f}  "
              f"{float(a['p_occ']):>7.2f} {float(d['p_occ']):>7.2f}  "
              f"{'YES' if sig else ('-' if flipped else '')}")
        out_rows.append({
            "stem": k[0], "z_m": k[1], "m_chi_GeV": k[2],
            "A_dom": da, "D_dom": dd, "flipped": int(flipped), "significant": int(sig),
            **{f"A_{c}": float(a[c]) for c in CHS},
            **{f"D_{c}": float(d[c]) for c in CHS},
        })

    with open(OUT / "mle_einc_bracketing.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        for r in out_rows: w.writerow(r)

    lines = [
        f"EXP-14: E_inc estimator robustness bracketing (default A vs oracle D)",
        f"  Both endpoints are computed; the Flow-I MLE will lie between.",
        f"",
        f"  Total cells: {len(shared)}",
        f"  Any flip: {len(flips)}/{len(shared)}",
        f"  Significant flip (max SIC > 1.0): {len(sig_flips)}/{len(shared)}",
        f"",
    ]
    if sig_flips:
        lines.append(f"  Significant flips:")
        for k, da, dd, a, d in sig_flips:
            lines.append(f"    {k[0]:<32s} z={float(k[1]):.2f} m={float(k[2]):g}  "
                         f"{da[2:]} -> {dd[2:]}  "
                         f"(A: {da[2:]}={float(a[da]):.2f},  D: {dd[2:]}={float(d[dd]):.2f})")
    else:
        lines.append(f"  No significant flips; conditioning convention is robust.")
    lines.append("")
    lines.append("Note: a full Flow-I gradient-inversion MLE is queued as a cluster experiment")
    lines.append("(see cluster/revision/mle_einc.job.yaml when prepared). The bracketing")
    lines.append("above is the laptop-side demonstration that the MLE answer is contained.")

    (OUT / "mle_einc_bracketing.txt").write_text("\n".join(lines) + "\n")
    print()
    print("\n".join(lines))
    print(f"\n[saved] {OUT}/mle_einc_bracketing.csv, _.txt")


# dispatch
STEPS = {
    "supervised_vs_unsupervised": supervised_vs_unsupervised,
    "aggregate_across_seeds": aggregate_across_seeds,
    "mechanism_map_multiseed": mechanism_map_multiseed,
    "matched_epsB_headline": matched_epsB_headline,
    "mle_einc_robustness": mle_einc_robustness,
}


def main():
    ap = argparse.ArgumentParser(description="Run Tier-1 result-producing analysis steps from committed inputs.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="run every step (default)")
    g.add_argument("--step", choices=list(STEPS), help="run a single step")
    args = ap.parse_args()
    if args.step:
        STEPS[args.step]()
    else:
        for name, fn in STEPS.items():
            print(f"\n########## STEP {name} ##########")
            fn()


if __name__ == "__main__":
    main()
