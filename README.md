# Calibrated Region-Resolved Anomaly Detection by Reusing a Trained Calorimeter Flow

Code and committed results for the paper of the same name, which builds calibrated,
factorized anomaly scores on top of a **trained** CaloFlow (Krause et al.,
[arXiv:2312.11618](https://arxiv.org/abs/2312.11618), Phys. Rev. D 110, 035036), with Flow-I
for the three layer energies and Flow-II for the 504-voxel morphology, reused without
retraining and without signal labels. Every figure and table regenerates from the committed
inputs here; no raw data, GPU, or cluster is needed.

## Reproduce

```bash
conda env create -f environment.yml && conda activate caloflow_ad   # Python 3.9
python figures.py  --all     # Figs 1(b)-10 → figures/
python tables.py   --all     # LaTeX rows of Tables I/III/IV/V/grid
python analysis.py --all     # derived-result post-processors
```

Each driver also takes `--fig`/`--table`/`--step` for a single target. The figure/table/
analysis layer needs only numpy, pandas, scipy, matplotlib; `torch`/`nflows` are in the env
only for the upstream provenance scans (`python -m pip install -r requirements.txt` is a pip
fallback). The committed `scan_outputs/` CSV/JSON tables are the canonical scores; the
GPU/cluster scans and per-event evaluations that produced them are provenance only and not in
this deposit (available on request).

## Result → script (function) → committed output

Generator filenames under `paper_build/v2/` are offset from the displayed figure numbers (a
documented remap); note **Fig 3 comes from `v2/fig4.py` and Fig 4 from `v2/fig5.py`**.

Figures (regenerate pixel-identical to the embedded PDFs):

| Displayed | `figures.py` | generator | committed input |
|---|---|---|---|
| Fig 1(a) | n/a (TikZ, PDF only) | `fig1_method_overview.tex` | shipped `figures/fig1_method_overview.pdf` |
| Fig 1(b) | `fig_eventpanel()` | `v2/fig2.py` | `data/in/event_{signal,photon}.npz` |
| Fig 2 | `fig2()` | `v2/fig3.py` | `data/in/coverage_outputs/*`, `E_cond.npy`, `achieved_vs_nominal.csv` |
| Fig 3 | `fig3()` | `v2/fig4.py` | `data/in/score_dist.npz` |
| Fig 4 | `fig4()` | `v2/fig5.py` | `scan_outputs/sic_grid_4ch.csv`, `data/in/maxregion_grid_seed001.csv` |
| Fig 5 | `fig5()` | `v2/fig_channelmass.py` | `scan_outputs/revision/{multiseed,typicality}/*.csv` |
| Fig 6 | `fig6()` | `v2/fig6.py` | `src/constants.py` (DEADZONE) |
| Fig 7 | `fig7()` | `v2/fig7.py` | `data/in/asymmetry.csv`, `data/in/fig7_region_sG_3seed.npz` |
| Fig 8 | `fig8()` | `v2/fig8.py` | `data/in/collinear_sic.csv` |
| Fig 9 | `fig9()` | `v2/fig9.py` | `scan_outputs/` contamination data |
| Fig 10 | `figS1()` | `v2/figS1.py` | `scan_outputs/achievability_table.csv`, occupancy ladder |

Tables (Tables II and VI are typeset inline in the manuscript and have **no** generator;
their values trace to the three-seed dead-zone medians and to
`scan_outputs/achievability_table.csv`/occupancy ladder respectively, and are not emitted by
`tables.py`):

| Table | `tables.py` | generator | committed input |
|---|---|---|---|
| I (background-null) | `table1()` | `v2/T1.py` | `src/constants.py`, `data/in/region_chi2.csv`, `data/in/logp_{total,I,II}.npy` |
| III (energy-tag leakage) | `table3()` | `v2/T2.py` | `src/constants.py`, `scan_outputs/revision/supervised_leakage.csv` |
| IV (oracle E_inc) | `table4()` | `v2/T_oracle.py` | `scan_outputs/E_inc_robustness_diff.txt` |
| V (architecture / split) | `table5()` | `v2/T_arch.py` | `data/splits/{c0,c1,c2,t}_idx_seed42.npy`, `Econd_bin_edges_5q_C0.npy` |
| grid (28×5 SIC; Table VII / Suppl.) | `table_grid()` | `v2/T3.py` | `scan_outputs/revision/multiseed/sic_grid_4ch_C0split_multiseed.csv` |

Derived-result steps (`analysis.py`, re-run byte-identical from committed data):

| Step | committed output |
|---|---|
| `supervised_vs_unsupervised` | `scan_outputs/supervised_vs_unsupervised.{csv,txt}` |
| `aggregate_across_seeds` | `scan_outputs/revision/multiseed/*_multiseed.csv`, `aggregate_summary.json` |
| `mechanism_map_multiseed` | `scan_outputs/revision/multiseed/mechanism_map_label_table.csv` |
| `matched_epsB_headline` | `scan_outputs/revision/multiseed/delta_sic_headline_*.{csv,txt}` |
| `mle_einc_robustness` | `scan_outputs/revision/<seed>/mle_einc_bracketing.csv` |

Headline numbers → committed source:

| Quantity (as quoted) | committed source |
|---|---|
| Factorization residual = 0 (10⁵ showers) | `data/in/logp_{total,I,II}.npy` (recomputed on CPU) |
| Per-region χ² nulls (286.5, 143.3, 74.0; core 15.74) | `data/in/region_chi2.csv`, `src/constants.py` (CHI2_LAYERS) |
| ‖z_II‖² per seed (503.85/505.98/495.61 vs 504) | `src/constants.py` (ZII_NORM_PER_SEED, D_II) |
| Flow-I off-shell z_I {159,108,262}→{2.08,2.09,2.04} | `src/constants.py` (ZI_NORM_*_PER_SEED) |
| 28×5 SIC grid, dead-zone p_occ 39.x, win T_occ | `scan_outputs/revision/multiseed/sic_grid_4ch_C0split_multiseed.csv` |
| Best-channel cells 31.6/28.8/30.2/40.7 (Fig 4) | `scan_outputs/sic_grid_4ch.csv`, `data/in/maxregion_grid_seed001.csv` |
| Oracle-E_inc robustness (0/28 flips, worst +3.13) | `scan_outputs/E_inc_robustness_diff.txt` |
| Energy-only controls (2.10/2.22; 2.54/14.71) | `scan_outputs/revision/supervised_leakage.csv` + constants |
| Mondrian split 20k/30k/20k/30k; edges −2.083…+0.007 | `data/splits/{c0,c1,c2,t}_idx_seed42.npy`, `Econd_bin_edges_5q_C0.npy` |

## Scope and provenance

Fig 4 prints per-cell max-region SIC for seed 001 (e.g. z=1.00 → 28.8), whereas Table II
reports the three-seed median (28.65); the full-grid three-seed-median max-region M is not
committed, so Fig 4 cannot be regenerated to three-seed medians from committed data. Fig 7's
input was relocated: the source computes the per-region s_G profile from ≈605 MB of per-seed
Flow-II latents, and `data/in/fig7_region_sG_3seed.npz` holds the identical reduced arrays the
figure plots (renders pixel-identical). Fig 1(a) is a hand-drawn TikZ schematic shipped as
`figures/fig1_method_overview.pdf` only; the `.tex` source is not in the tree.

The three drivers plus `src/` (`style`, `constants`, `data_io`, `calo_grid`) are
laptop-runnable from committed inputs. The GPU/cluster provenance scans and per-event
evaluations that produced `scan_outputs/` are not included in this deposit. The dataset
(Zenodo [`10393540`](https://zenodo.org/records/10393540): `gamma_1.hdf5` train/val,
`gamma_2.hdf5` calibration/test, displaced-signal files) and the trained CaloFlow
(<https://gitlab.com/claudius-krause/caloflow>) are likewise not committed. Results use three
independently trained checkpoints (seed_001/002/003); the dataset split is frozen once with
seed 42 and never re-split. `docs/sha256_manifest.txt` pins the byte-reproducible committed
inputs/code; the shipped figure PDFs are verified by pixel comparison, not by hash (matplotlib
metadata varies). The exact code state is the final-submission git commit.
