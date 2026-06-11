#!/usr/bin/env python3
"""Figure generator: one function per displayed figure, each regenerating from
committed inputs in this repo.

Displayed figure  ->  source generator (paper_build/v2/)  ->  embedded basename
  Fig 1(a) schematic   (TikZ figures/fig1_method_overview.tex; NOT in this file)
  Fig 1(b) event panel   fig2.py             -> fig_eventpanel.pdf
  Fig 2  calibration     fig3.py             -> fig2.pdf
  Fig 3  score dists     fig4.py             -> fig3.pdf
  Fig 4  best-channel    fig5.py             -> fig4.pdf      (seed 001 per-cell)
  Fig 5  mechanism mass  fig_channelmass.py  -> fig5.pdf
  Fig 6  ...             fig6.py             -> fig6.pdf
  Fig 7  latent regions  fig7.py             -> fig7.pdf
  Fig 8  collinear       fig8.py             -> fig8.pdf
  Fig 9  contamination   fig9.py             -> fig9.pdf
  Fig 10 occupancy       figS1.py            -> figS1.pdf
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import style as S        # noqa: E402
import constants as C    # noqa: E402
import data_io as D      # noqa: E402  (used by several figures)
import calo_grid as G    # noqa: E402  (used by the event-panel figure)

S.apply_style(use_tex=False)
OUT = REPO / "figures"


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[saved]", OUT / f"{name}.pdf", "+ .png")


# Fig 4 <- paper_build/v2/fig5.py (best-channel SIC heatmap; seed 001).
# Reads scan_outputs/sic_grid_4ch.csv + data/in/maxregion_grid_seed001.csv.
def fig4():
    CSV = REPO / "scan_outputs" / "sic_grid_4ch.csv"
    MGRID = REPO / "data" / "in" / "maxregion_grid_seed001.csv"   # relocated from paper_build/

    COMPETITORS = [
        ("raw_LL", "T_raw"),
        ("M_sic", "M"),
        ("p_occ", "T_occ"),
        ("p_comb", "p_comb"),
    ]
    SILENT_SIC = 0.1                  # below this a cell is "silent" (achievability floor)
    VMIN, VMAX = SILENT_SIC, C.SIC_SATURATION   # viridis range: 0.1 .. 31.62 (1/sqrt(alpha))

    def _load():
        df = pd.read_csv(CSV)
        mg = pd.read_csv(MGRID)
        df = df.merge(mg[["z_m", "m_chi_GeV", "M_sic"]], on=["z_m", "m_chi_GeV"], how="left")
        # Documented z=1.24 morphology-latent truncation: HARD-set M -> NaN.
        df.loc[df["z_m"] == 1.24, "M_sic"] = np.nan
        return df

    def _grids(df, masses, zs):
        """Return (winner_sic, winner_key, m_truncated) 2-D arrays over (mass, z)."""
        cols = [c for c, _ in COMPETITORS]
        sic = np.full((len(masses), len(zs)), np.nan)
        win = np.empty((len(masses), len(zs)), dtype=object)
        mtrunc = np.zeros((len(masses), len(zs)), dtype=bool)
        for i, mm in enumerate(masses):
            for j, z in enumerate(zs):
                r = df[(df.z_m == z) & (df.m_chi_GeV == mm)]
                if r.empty:
                    continue
                r = r.iloc[0]
                # z=1.24 M is truncated (NaN); flag the row so the M omission is visible.
                mtrunc[i, j] = bool(pd.isna(r["M_sic"])) and (z == 1.24)
                finite = {key: r[col] for col, key in COMPETITORS if np.isfinite(r[col])}
                best = max(finite, key=finite.get)
                sic[i, j] = finite[best]
                win[i, j] = best
        return sic, win, mtrunc

    df = _load()
    masses = [5, 0.5, 0.05, 0.005]          # log-spaced, high -> low (top -> bottom)
    zs = list(C.GEOMETRY["z_m"])            # 0.33 ... 1.24
    sic, win, mtrunc = _grids(df, masses, zs)
    nz, nm = len(zs), len(masses)

    norm = LogNorm(vmin=VMIN, vmax=VMAX)
    cmap = mpl.colormaps[S.CMAP_SIC].copy()   # viridis -- colorful heatmap theme
    NEUTRAL = (0.92, 0.92, 0.92, 1.0)         # silent-cell neutral fill

    fig, ax = plt.subplots(figsize=(S.COL_FULL, 4.0))

    # Continuous viridis heatmap: color = SIC of the best channel at each cell
    # (purple -> green -> yellow toward the 1/sqrt(alpha) ceiling). Silent cells
    # (below the achievability floor) are masked and drawn neutral grey.
    disp = np.ma.masked_where(~np.isfinite(sic) | (sic < SILENT_SIC),
                              np.clip(sic, VMIN, VMAX))
    im = ax.imshow(disp, cmap=cmap, norm=norm, aspect="auto", origin="upper")

    for i in range(nm):
        for j in range(nz):
            v = sic[i, j]
            silent = (not np.isfinite(v)) or (v < SILENT_SIC)
            if silent:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=NEUTRAL,
                                       edgecolor="none", zorder=1))
                ax.text(j, i, "silent", ha="center", va="center", color="0.55",
                        fontsize=6.5, zorder=3)
                continue
            face = cmap(norm(np.clip(v, VMIN, VMAX)))
            tc = "white" if S._luminance(face) < 0.6 else "black"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", color=tc,
                    fontsize=9.5, fontweight="bold", zorder=3)
            if mtrunc[i, j]:                 # M not evaluable here (truncated latent)
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       hatch="////", edgecolor="white", lw=0.0,
                                       zorder=2))

    ax.set_xticks(range(nz)); ax.set_xticklabels([f"{z:.2f}" for z in zs])
    ax.set_yticks(range(nm)); ax.set_yticklabels([f"{m:g}" for m in masses])
    ax.set_xlabel(r"displacement $z$ [m]")
    ax.set_ylabel(r"dark-matter mass $m_\chi$ [GeV]")
    S.no_minor_x(ax)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_xlim(-0.5, nz - 0.5); ax.set_ylim(nm - 0.5, -0.5)

    # Colorbar: best-channel SIC, with the 1/sqrt(alpha) ceiling marked at the top.
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, aspect=28, extend="min")
    cb.set_label(r"best-channel SIC $=\varepsilon_S/\sqrt{\varepsilon_B}$  ($\alpha=10^{-3}$)")
    cb.ax.axhline(C.SIC_SATURATION, color="black", ls=":", lw=1.0)
    cb.ax.text(1.7, C.SIC_SATURATION, r"$1/\sqrt{\alpha}=%.1f$" % C.SIC_SATURATION,
               transform=cb.ax.get_yaxis_transform(), ha="left", va="center",
               fontsize=7, rotation=90)

    # Compact 2-entry key ABOVE the heatmap (horizontal) so the right margin
    # holds only the colorbar -- removes the big legend gap.
    handles = [Rectangle((0, 0), 1, 1, facecolor=NEUTRAL, edgecolor="0.7",
                         label=r"silent (SIC $<0.1$)"),
               Rectangle((0, 0), 1, 1, fill=False, hatch="////", edgecolor="0.45",
                         label=r"$M$ truncated ($z{=}1.24$)")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.005),
              ncol=2, frameon=False, fontsize=8, handlelength=1.4,
              handleheight=1.3, columnspacing=2.0, borderaxespad=0.0)

    fig.subplots_adjust(left=0.09, right=0.92, top=0.88, bottom=0.13)
    _save(fig, "fig4")


# fig_eventpanel (ported from paper_build/v2/)
def fig_eventpanel():
    from matplotlib.ticker import MaxNLocator

    def G_data(kind):
        """Load committed event NPZ via data_io (no synthesized demo)."""
        return D.load_event(kind, demo=False)

    def build():
        sig = G.load_shower(G_data("signal"))
        pho = G.load_shower(G_data("photon"))

        # Shared log-energy normalization across all six layer images.
        norm = G.energy_norm([sig["layer0"], sig["layer1"], sig["layer2"],
                              pho["layer0"], pho["layer1"], pho["layer2"]], log_energy=True)

        fig = plt.figure(figsize=(S.COL_FULL, 3.9))
        gs = fig.add_gridspec(2, 3, hspace=0.08, wspace=0.10)
        ax_sig = [fig.add_subplot(gs[0, j]) for j in range(3)]
        ax_pho = [fig.add_subplot(gs[1, j]) for j in range(3)]

        # Viridis energy colormap for both rows -- matches the parent paper's
        # (Krause et al.) event displays and the clean colorful reference figure.
        G.draw_layers(ax_sig, sig, norm=norm, colorbar=False, hide_ticks=False,
                      cmap=G.CMAP_IMAGE)
        ims, _ = G.draw_layers(ax_pho, pho, norm=norm, colorbar=False, hide_ticks=False,
                               cmap=G.CMAP_IMAGE)

        # No overlays: the panel orients by raw morphology alone -- signal (a) shows
        # two energy blobs, prompt photon (b) a single core. All-or-nothing labeling;
        # a lone core box re-opened the "core between the two photons" ambiguity, so
        # we annotate neither and let the caption state the contrast.

        for ax in ax_pho:                       # layer titles only on the top (signal) row
            ax.set_title("")
        for ax in ax_pho:                       # eta Cell ID on the bottom (shared) row
            ax.set_xlabel(r"$\eta$ Cell ID", fontsize=8)
        for ax in ax_sig:                       # rows share the eta axis -> hide the
            ax.tick_params(labelbottom=False)   # redundant x-numbers on the top row
        for ax in ax_sig + ax_pho:              # integer cell-ID ticks (no half-integers)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.tick_params(labelsize=7)
            S.no_minor_x(ax)                    # categorical cell-ID axis: no notch minor ticks

        cb = fig.colorbar(ims[-1], ax=ax_sig + ax_pho, fraction=0.028, pad=0.02)
        cb.set_label("Energy (MeV)")

        ax_sig[0].set_ylabel(r"signal 5 GeV, $z=0.33$" + "\n" + r"$\phi$ Cell ID",
                             fontsize=8, fontweight="bold")
        ax_pho[0].set_ylabel("prompt photon\n" + r"$\phi$ Cell ID",
                             fontsize=8, fontweight="bold")

        S.add_panel_label(ax_sig[0], "a")
        S.add_panel_label(ax_pho[0], "b")
        return fig

    fig = build()
    _save(fig, "fig_eventpanel")


# fig2 (ported from paper_build/v2/)
def fig2():
    from scipy.stats import beta

    CO = REPO / "data" / "in" / "coverage_outputs"
    N_TEST = 50_000  # background test count behind every marginal achieved-eps_B point

    # calibration object for both panels = Flow-II scalar p-value (max-region M family)
    CH = "M"
    COL = S.CHANNELS[CH]["color"]

    def clopper_pearson(k, n, cl=0.6827):
        """Two-sided Clopper-Pearson interval at confidence level cl (default 1-sigma)."""
        k = np.asarray(k, float)
        n = float(n)
        a = 1.0 - cl
        lo = np.where(k > 0, beta.ppf(a / 2, k, n - k + 1), 0.0)
        hi = np.where(k < n, beta.ppf(1 - a / 2, k + 1, n - k), 1.0)
        return lo, hi

    # Panel (a): marginal achieved-vs-nominal with Clopper-Pearson band.
    def _marginal():
        df = pd.read_csv(REPO / "data" / "in" / "achieved_vs_nominal.csv")
        nominal = df["nominal"].to_numpy()
        achieved = df["achieved"].to_numpy()
        order = np.argsort(nominal)
        nominal, achieved = nominal[order], achieved[order]
        k = np.round(achieved * N_TEST).astype(int)  # binomial successes behind each point
        lo, hi = clopper_pearson(k, N_TEST)
        return nominal, achieved, lo, hi

    # Panel (b): per-bin conditional coverage at alpha=1e-3, in the production
    # 5-quantile Mondrian binning. Test p-values are re-binned by re-deriving each
    # event's log10(E_cond) and assigning it with the committed 5-quantile edges.
    # ALIGNMENT VERIFIED: this exact procedure with the 10-bin edges reproduces the
    # committed bins_test.npy for 100% of events, confirming p_S <-> test_idx <->
    # E_cond are aligned.
    def _per_bin(alpha=C.ALPHA):
        pS = np.load(CO / "p_S.npy")
        tidx = np.load(CO / "test_idx.npy")
        Econd = np.load(REPO / "data" / "in" / "E_cond.npy")
        edges = np.load(REPO / "data" / "splits" / "Econd_bin_edges_5q_C0.npy")
        q = np.log10(Econd[tidx])
        bt = np.clip(np.digitize(q, edges) - 1, 0, len(edges) - 2)
        nb = len(edges) - 1
        ach, blo, bhi, floor = [], [], [], []
        for b in range(nb):
            m = bt == b
            n = int(m.sum())
            kk = int((pS[m] <= alpha).sum())
            cl, ch = clopper_pearson(kk, n)
            ach.append(kk / n)
            blo.append(float(cl))
            bhi.append(float(ch))
            floor.append(np.floor(alpha * (n + 1)) / (n + 1))
        return (np.array(ach), np.array(blo), np.array(bhi), np.array(floor), nb)

    fig, ax = plt.subplots(1, 2, figsize=(S.COL_FULL, 3.1))

    # (a) marginal
    nom, ach, lo, hi = _marginal()
    axa = ax[0]
    flo = min(nom.min(), ach.min()) * 0.7  # crop lower range to just below the data
    S.identity_diagonal(axa, flo, 1.0)
    axa.fill_between(nom, lo, hi, color=COL, alpha=0.25, lw=0)
    axa.plot(nom, ach, marker=S.CHANNELS[CH]["marker"], ms=3.2, lw=1.1,
             color=COL, label=r"achieved $\varepsilon_B$ (68% band)")
    axa.set_title("overall", fontsize=9)
    axa.set_xscale("log")
    axa.set_yscale("log")
    axa.set_xlim(flo, 1.0)
    axa.set_ylim(flo, 1.0)
    axa.set_xlabel(r"nominal level $\alpha$")
    axa.set_ylabel(r"achieved FPR $\varepsilon_B$")
    # On-panel number annotations removed: the diagonal IS the message; the
    # mean p=0.4997 and achieved eps_B=9.6e-4 @ alpha=1e-3 are stated in the text
    # (Sec.~V.A) and need no leader-arrow clutter here.
    axa.legend(loc="upper left", fontsize=7.0, frameon=False, handlelength=1.4)
    S.despine(axa)
    S.add_panel_label(axa, "a")

    # (b) conditional per-bin
    pach, blo, bhi, _floor, nb = _per_bin()
    axb = ax[1]
    x = np.arange(nb)  # categorical bin positions (5 production Mondrian bins)
    axb.axhline(C.ALPHA, color="black", ls="--", lw=1.1,
                label=r"nominal $\alpha=10^{-3}$")
    # (achievability floor omitted: at ~10k events/bin it sits on the nominal
    #  line, so it is redundant here; it is load-bearing only for the discrete
    #  occupancy channel, Fig.~10.)
    yerr = np.vstack([pach - blo, bhi - pach])
    axb.errorbar(x, pach, yerr=yerr, fmt=S.CHANNELS[CH]["marker"], ms=4.0,
                 color=COL, capsize=2.2, lw=0.9, mew=0.8,
                 label="achieved (per bin)")
    axb.set_yscale("log")
    axb.set_ylim(2e-4, 5e-3)
    axb.set_yticks([3e-4, 1e-3, 3e-3])
    axb.set_yticklabels([r"$3\times10^{-4}$", r"$10^{-3}$", r"$3\times10^{-3}$"])
    axb.set_xticks(x)
    axb.set_xticklabels([str(i + 1) for i in range(nb)])
    axb.set_xlim(-0.6, nb - 0.4)
    axb.set_title("per energy bin", fontsize=9)
    axb.set_xlabel(r"Mondrian $\log_{10}(E_{\rm inc})$ bin")
    axb.set_ylabel(r"achieved $\varepsilon_B$ @ $\alpha=10^{-3}$")
    S.no_minor_x(axb)
    axb.legend(loc="upper right", fontsize=6.2, frameon=False, ncol=1,
               handlelength=1.4)
    S.despine(axb)
    S.add_panel_label(axb, "b")

    fig.tight_layout(w_pad=2.0)
    _save(fig, "fig2")


# fig3 (ported from paper_build/v2/)
def fig3():
    def _density_hist(ax, x, edges, name, *, filled):
        """Unit-area density histogram; filled = translucent fill, else stepped outline."""
        col = S.CHANNELS[name]["color"]
        counts, _ = np.histogram(x, bins=edges)
        widths = np.diff(edges)
        norm = counts.sum() * widths            # unit-area (density) normalization
        centers = 0.5 * (edges[:-1] + edges[1:])
        dens = counts / norm
        if filled:
            ax.fill_between(centers, dens, step="mid", color=col, alpha=0.45,
                            lw=0, zorder=1)
            ax.step(centers, dens, where="mid", color=col, lw=0.9, zorder=2)
        else:
            ax.step(centers, dens, where="mid", color=col, lw=1.6, zorder=3)

    def _panel(ax, bg, sig, edges, xlabel):
        # background = prompt photon (frozen T_raw gray for the raw panel, M hue for M)
        _density_hist(ax, bg, edges, "T_raw", filled=True)
        _density_hist(ax, sig, edges, "M", filled=False)
        ax.set_yscale("log")
        ax.set_ylim(top=ax.get_ylim()[1] * 30)  # headroom so the legend floats above all curves
        ax.set_xlabel(xlabel)
        ax.set_ylabel("normalized density")
        S.despine(ax)

    d = np.load(REPO / "data" / "in" / "score_dist.npz")
    sic_traw = C.DEADZONE["z=1.00"]["T_raw"][0]   # 0.07 (committed)
    sic_m = C.DEADZONE["z=1.00"]["M"][0]          # 28.65 (committed)

    fig, ax = plt.subplots(1, 2, figsize=(S.FIG_W * 0.72, 2.9))

    # legend proxies (frozen colors): gray fill = background, vermillion line = signal
    bg_h = plt.Rectangle((0, 0), 1, 1, fc=S.CHANNELS["T_raw"]["color"], alpha=0.45,
                         ec="none")
    sig_h = plt.Line2D([0], [0], color=S.CHANNELS["M"]["color"], lw=1.6)

    # (a) raw scalar score: near-complete overlap -> "why decompose"
    _panel(ax[0], d["Traw_photon"], d["Traw_signal"],
           np.linspace(-200, 300, 48),
           r"raw shower $\log p_{\rm tot}$  (nats)")
    ax[0].set_title(r"raw scalar score $T_{\rm raw}$  (silent)", fontsize=9, pad=6)
    ax[0].text(0.04, 0.04, r"SIC$(T_{\rm raw})=%.2f$" % sic_traw,
               transform=ax[0].transAxes, fontsize=8.5, va="bottom", ha="left",
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", lw=0.7))
    ax[0].legend([bg_h, sig_h], ["photon background", "dead-zone signal"],
                 loc="upper right", fontsize=7.5, frameon=True, framealpha=0.9,
                 facecolor="white", edgecolor="0.8")

    # (b) max-region score: clean separation (the recovery preview)
    _panel(ax[1], d["M_photon"], d["M_signal"],
           np.linspace(0, 40, 48),
           r"max-region statistic $M$")
    ax[1].set_title(r"max-region score $M$  (separates)", fontsize=9, pad=6)
    # SIC box in the genuinely empty lower-right (M>20, density<1e-3: no curve there)
    ax[1].text(0.97, 0.04, r"SIC$(M)=%.1f$" % sic_m,
               transform=ax[1].transAxes, fontsize=8.5, va="bottom", ha="right",
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", lw=0.7))
    ax[1].legend([bg_h, sig_h], ["photon background", "dead-zone signal"],
                 loc="upper right", fontsize=7.5, frameon=True, framealpha=0.9,
                 facecolor="white", edgecolor="0.8")

    S.add_panel_label(ax[0], "a")
    S.add_panel_label(ax[1], "b")
    _save(fig, "fig3")


# fig5 (ported from paper_build/v2/)
def fig5():
    from matplotlib.lines import Line2D

    sm = pd.read_csv(REPO / "scan_outputs/revision/typicality/summary_median.csv")
    ms = pd.read_csv(REPO / "scan_outputs/revision/multiseed/sic_grid_4ch_C0split_multiseed.csv")

    # Mass grid (GeV), log-spaced, ascending.
    MASSES = np.array([5e-3, 5e-2, 5e-1, 5.0])

    # Facet z slices: z=1.16 (both families active -> the clean aggregate-vs-
    # morphology contrast) and z=1.24 (late-start, aggregate-dominant). z=0.33
    # (aggregate silent at zero) and z=1.00 (dead zone, all silent) are dropped --
    # they showed only partial pictures and obscured the message.
    Z_FACETS = [1.16, 1.24]
    REGIME = {1.16: "both families active", 1.24: "late-start: aggregate dominant"}

    def _series(df, zval, value_cols):
        """Return arrays aligned to MASSES for the given z slice and value columns."""
        sub = df[np.isclose(df["z_m"], zval)]
        out = []
        for col in value_cols:
            vals = []
            for m in MASSES:
                row = sub[np.isclose(sub["m_chi_GeV"], m)]
                vals.append(float(row[col].iloc[0]) if len(row) else np.nan)
            out.append(np.array(vals))
        return out

    fig, axes = plt.subplots(1, 2, figsize=(S.COL_FULL, 3.7), sharex=True, sharey=False)
    panel_letters = ["a", "b"]

    # Aggregate channels render SOLID, morphology DASHED (use_family_ls=True).
    # T_I, T_occ  = aggregate ; T_S = morphology ; T_raw = scalar (also dashed).
    for ip, (ax, zval) in enumerate(zip(axes, Z_FACETS)):
        # --- T_raw (scalar, mixed; collapses) -- 3-seed band ---
        traw_med, traw_lo, traw_hi = _series(
            sm, zval, ["SIC_T_raw_med", "SIC_T_raw_min", "SIC_T_raw_max"])
        S.seed_band(ax, MASSES, traw_med, traw_lo, traw_hi, "T_raw",
                    use_family_ls=True, label=(ip == 0), markersize=4)

        # --- T_I (Flow-I aggregate typicality; mass-independent). One of three
        #     seeds is inactive (documented Flow-I latent instability), so the
        #     min-max spread is wide -> render it as thin whiskers, NOT a filled
        #     band, so the median (the flat, mass-independent story) stays readable. ---
        ti_med, ti_lo, ti_hi = _series(
            sm, zval, ["SIC_T_I_med", "SIC_T_I_min", "SIC_T_I_max"])
        S.plot_channel(ax, MASSES, ti_med, "T_I", use_family_ls=True,
                       label=(ip == 0), markersize=4)
        ax.errorbar(MASSES, ti_med, yerr=[ti_med - ti_lo, ti_hi - ti_med],
                    fmt="none", ecolor=S.CHANNELS["T_I"]["color"], elinewidth=0.6,
                    capsize=0, alpha=0.3, zorder=2)

        # --- T_S (Flow-II morphology scalar; collapses) -- 3-seed band ---
        ts_med, ts_lo, ts_hi = _series(
            sm, zval, ["SIC_T_S_med", "SIC_T_S_min", "SIC_T_S_max"])
        S.seed_band(ax, MASSES, ts_med, ts_lo, ts_hi, "T_S",
                    use_family_ls=True, label=(ip == 0), markersize=4)

        # --- T_occ (occupancy aggregate; deterministic on raw HDF5 -> no band).
        #     Plotted at its true value; the z=1.24 panel's y-axis extends to fit
        #     ~40, which exceeds the 1/sqrt(alpha) ceiling because the discrete
        #     occupancy score sits below nominal alpha at its achievability floor. ---
        (tocc,) = _series(ms, zval, ["p_occ_mean"])
        S.plot_channel(ax, MASSES, tocc, "T_occ", use_family_ls=True,
                       label=(ip == 0), markersize=4)
        # (max-region M is not shown here -- it is the combined envelope, the headline
        #  of the mechanism map / recovery figures; this figure isolates the two
        #  decomposition families.)

        # axes config -- independent y per panel: the z=1.24 occupancy (~40) fits
        # without capping, while z=1.16 still fills its range.
        ax.set_xscale("log")
        ax.set_xlim(3e-3, 8.0)
        ax.set_ylim(-1.0, 16.0 if zval < 1.2 else 43.0)
        S.despine(ax)
        S.grid_y(ax)
        S.add_panel_label(ax, panel_letters[ip])
        ax.set_title(rf"$z={zval:.2f}\,$m — {REGIME[zval]}", fontsize=9)
        ax.set_xlabel(r"dark-matter mass $m_\chi$ (GeV)")
        if ip == 0:
            ax.set_ylabel(r"per-channel SIC $=\varepsilon_S/\sqrt{\varepsilon_B}$")
        # 1/sqrt(alpha) ceiling reference (visible only in the z=1.24 panel, where
        # T_occ exceeds it); labelled there.
        cv = S.sic_ceiling(ax, label=False)
        if zval >= 1.2:
            ax.text(3.3e-3, cv, r"$1/\sqrt{\alpha}=31.6$", ha="left", va="bottom",
                    fontsize=7.5)

    # One family-tagged legend. T_raw = T_I + T_S is the scalar mixture; in these
    # cells its variation is morphology-dominated, so it is labeled "scalar,
    # morph.-dom." Aggregate = solid, morphology / morph.-dominated = dashed.
    def _h(name, ls, fam):
        m = S.CHANNELS[name]
        return Line2D([0], [0], color=m["color"], marker=m["marker"], ls=ls, ms=5,
                      lw=1.6, label=rf"{m['label']} ({fam})")

    fig.legend(handles=[_h("T_I", "-", "aggregate"), _h("T_occ", "-", "aggregate"),
                        _h("T_S", "--", "morphology"), _h("T_raw", "--", "scalar, morph.-dom.")],
               loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
               fontsize=8.5, handlelength=2.2, columnspacing=2.4, labelspacing=0.55)

    _save(fig, "fig5")


# fig6 (ported from paper_build/v2/)
def fig6():
    # Four cells (5 GeV), display order matching the reference original.
    CELLS = ["z=0.33", "z=1.00", "z=1.04", "z=1.16"]
    # Four channels per cell, in the frozen channel vocabulary.
    CH = ["T_raw", "global", "M", "supervised"]

    def build():
        fig, ax = plt.subplots(figsize=(S.FIG_W, 3.6))

        x = np.arange(len(CELLS), dtype=float)
        n = len(CH)
        w = 0.20                                   # bar width
        # symmetric group offsets: (i - (n-1)/2) * w
        offsets = (np.arange(n) - (n - 1) / 2.0) * w

        for i, name in enumerate(CH):
            st = S.CHANNELS[name]
            meds, lo_err, hi_err = [], [], []
            for cell in CELLS:
                m, (le, he) = C.med_err(C.DEADZONE[cell][name])
                meds.append(m)
                lo_err.append(le)
                hi_err.append(he)
            ax.bar(x + offsets[i], meds, w, yerr=[lo_err, hi_err],
                   color=st["color"], edgecolor="black", lw=0.5,
                   error_kw=dict(elinewidth=1.0, capsize=2.5, capthick=0.9,
                                 ecolor="black"),
                   label=st["label"], zorder=3)

        # 1/sqrt(alpha) achievability ceiling, labelled directly on the line (kept
        # out of the legend) at the right end.
        ax.axhline(C.SIC_SATURATION, color="black", ls=":", lw=1.0, zorder=2)
        ax.text(len(CELLS) - 0.45, C.SIC_SATURATION + 0.25,
                r"$1/\sqrt{\alpha}=31.6$", ha="right", va="bottom", fontsize=8)

        # recovery ratio rho printed above each cell group.
        for j, cell in enumerate(CELLS):
            r = C.DEADZONE[cell]["rho"][0]
            ax.text(x[j], 33.0, r"$\rho=%.2f$" % r, ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("z=", r"$z=$") + "\n5 GeV" for c in CELLS])
        ax.set_xlim(-0.6, len(CELLS) - 0.4)
        ax.set_ylim(0, 36)
        ax.set_ylabel(r"SIC $=\varepsilon_S/\sqrt{\varepsilon_B}$ at $\alpha=10^{-3}$")
        ax.set_xlabel("dead-zone cell (shower depth $z$)")
        S.no_minor_x(ax)
        S.despine(ax)
        S.grid_y(ax)
        # Channel key as a horizontal strip ABOVE the bars (outside the axes) so it
        # never competes with the tall bars for space.
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=4,
                  fontsize=8.5, columnspacing=1.6, handletextpad=0.5,
                  frameon=False, borderaxespad=0.0)
        return fig

    fig = build()
    _save(fig, "fig6")


# fig7 (ported from paper_build/v2/)
def fig7():
    from matplotlib.lines import Line2D

    def panel_scatter(ax):
        df = pd.read_csv(REPO / "data" / "in" / "asymmetry.csv")
        sig = df[df.kind == "signal"]
        pho = df[df.kind == "photon"]
        c_sig = S.CHANNELS["M"]["color"]
        c_pho = "#6b6b6b"   # visible medium gray (the frozen photon #b0b0b0 washes out here)

        ax.axhline(0, color="0.8", lw=0.7, zorder=0)
        ax.axvline(0, color="0.8", lw=0.7, zorder=0)

        # Two clouds, read directly: the photon-null CONTROL (prompt photons, no
        # signal) is a round gray cloud at the origin -- no left/right correlation;
        # the signal is a vermillion cloud tilted along the anti-diagonal (latent
        # excess sits opposite the depleted core). Drawn signal-on-top.
        ax.scatter(sig.raw_asym, sig.latent_asym, s=2.5, alpha=0.05, zorder=1,
                   color=c_sig, edgecolors="none", rasterized=True)
        # photon-null on top and clearly visible: darker gray, higher opacity so the
        # flat control cloud reads against the vermillion signal.
        ax.scatter(pho.raw_asym, pho.latent_asym, s=4.0, alpha=0.18, zorder=3,
                   color=c_pho, edgecolors="none", rasterized=True)

        # signal OLS fit (the anti-correlation) -- dashed = derived line.
        b, a = np.polyfit(sig.raw_asym, sig.latent_asym, 1)
        xx = np.array([sig.raw_asym.min(), sig.raw_asym.max()])
        ax.plot(xx, a + b * xx, color="0.15", ls="--", lw=1.6, zorder=5)

        ax.set_xlabel("raw-image L/R asymmetry")
        ax.set_ylabel("latent excess L/R asymmetry")
        ax.set_ylim(-58, 58)

        # full-opacity legend swatches (the alpha-blended clouds are faint); labels
        # carry the reading so no in-panel text block is needed.
        handles = [
            Line2D([0], [0], marker="o", linestyle="none", markersize=6,
                   markerfacecolor=c_sig, markeredgecolor="none",
                   label="signal (anti-correlated)"),
            Line2D([0], [0], marker="o", linestyle="none", markersize=6,
                   markerfacecolor=c_pho, markeredgecolor="0.5",
                   label="photon-null control (flat)"),
            Line2D([0], [0], color="0.15", ls="--", lw=1.6, label="signal fit"),
        ]
        ax.legend(handles=handles, loc="upper right", handletextpad=0.5,
                  fontsize=7.5, borderpad=0.5)
        S.despine(ax)
        S.add_panel_label(ax, "a")

    def _region_3seed():
        """3-seed per-region mean s_G, read from the committed npz; returns ordered
        (keys, median, lo, hi)."""
        d = np.load(REPO / "data" / "in" / "fig7_region_sG_3seed.npz", allow_pickle=True)
        return list(d["order"]), d["med"], d["lo"], d["hi"]

    def panel_regions(ax):
        keys, med, lo, hi = _region_3seed()
        disp = {"L0": "L0", "L1": "L1 (all)", "L2": "L2", "core": "L1 core",
                "mid": "L1 mid", "outer": "L1 outer", "left": "L1 left",
                "right": "L1 right"}
        labels = [disp[k] for k in keys]
        c_core = S.CHANNELS["M"]["color"]
        c_other = S.CHANNELS["global"]["color"]
        colors = [c_core if k == "core" else c_other for k in keys]
        n_layer = 3                                    # L0,L1,L2 then a visual gap
        xpos = np.array([i if i < n_layer else i + 0.7 for i in range(len(keys))])
        yerr = np.vstack([med - lo, hi - med])         # 3-seed [min,max] band

        ax.bar(xpos, med, color=colors, edgecolor="black", lw=0.5, yerr=yerr,
               error_kw=dict(elinewidth=1.0, capsize=2.5, capthick=0.9, ecolor="black"))
        ax.axhline(0, color="0.8", lw=0.7)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel(r"mean $s_G$  (z = 0.33, 5 GeV)")
        ax.set_ylim(0, hi.max() * 1.15)

        handles = [
            Line2D([0], [0], marker="s", linestyle="none", markersize=7,
                   markerfacecolor=c_core, markeredgecolor="black",
                   label="L1 core (depleted)"),
            Line2D([0], [0], marker="s", linestyle="none", markersize=7,
                   markerfacecolor=c_other, markeredgecolor="black",
                   label="other region"),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=7.5, handletextpad=0.4)
        ax.tick_params(axis="x", which="both", length=0)  # 3-seed median; bars = [min,max]
        S.despine(ax)
        S.add_panel_label(ax, "b")

    def build():
        fig, ax = plt.subplots(1, 2, figsize=(S.COL_FULL, 3.1))
        panel_scatter(ax[0])
        panel_regions(ax[1])
        fig.tight_layout()
        return fig

    fig = build()
    _save(fig, "fig7")


# fig8 (ported from paper_build/v2/)
def fig8():
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # Data: data/in/collinear_sic.csv (cols ratio, sic, supervised_auc).
    # 4 chi->gamma gamma mass points; ratio = Dx_kin/Dx_resolve grows with mass.
    df = D.load_collinear_sic(False).sort_values("ratio").reset_index(drop=True)
    # mass label per row in ascending-ratio order (5 MeV ... 5 GeV)
    MASSES = ["5 MeV", "50 MeV", "500 MeV", "5 GeV"]

    sic = np.clip(df.sic.to_numpy(), 1e-2, None)
    ratio = df.ratio.to_numpy()

    fig, ax = plt.subplots(figsize=(S.COL_SINGLE * 1.95, 4.4))

    # --- Main: morphology T_S SIC vs ratio (frozen T_S color+marker, dashed family)
    (curve,) = S.plot_channel(ax, ratio, sic, "T_S", use_family_ls=True, ms=7, lw=1.6,
                              label=r"morphology $T_S$ ($\chi\!\to\!\gamma\gamma$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta x_{\rm kin}/\Delta x_{\rm resolve}$")
    ax.set_ylabel(r"$T_S$  SIC")
    # widen limits so labels and annotations have empty margin to live in
    ax.set_xlim(1.3e-3, 9.0)
    ax.set_ylim(4e-3, 110.0)

    # Unresolved region (ratio < 1): the two prongs fall inside one cell. Shade it
    # lightly and draw the boundary line -- this carries the physics VISUALLY so no
    # prose sits on the axes (the "resolution limit, not a method failure" reading
    # belongs in the caption).
    ax.axvspan(1.3e-3, 1.0, color="0.5", alpha=0.08, lw=0, zorder=0)
    S.floor_line(ax, 1.0, orient="v")          # boundary line only (no rotated text)

    # Mass labels: data labels pushed straight UP off each marker into empty vertical
    # space so no label sits on the curve or another label.
    tcol = S.CHANNELS["T_S"]["color"]
    offs = [(0, 11), (0, 11), (0, 11), (-4, 13)]
    has = ["center", "center", "center", "right"]
    for r, s, m, off, ha in zip(ratio, sic, MASSES, offs, has):
        ax.annotate(m, (r, s), textcoords="offset points", xytext=off,
                    fontsize=8, ha=ha, va="bottom", color=tcol, fontweight="bold")

    S.despine(ax)
    S.grid_y(ax)

    # Self-identifying legend (replaces all on-axes prose): the curve, the boundary,
    # and the shaded unresolved zone. Interpretation lives in the caption.
    ax.legend(handles=[
        curve,
        Line2D([0], [0], color="0.55", ls=(0, (1, 1)), lw=1.0, label="resolvable boundary"),
        Patch(facecolor="0.5", alpha=0.20, lw=0, label="unresolved (prongs merge)"),
    ], loc="upper left", fontsize=8.5, handlelength=2.2, borderaxespad=0.7,
       labelspacing=0.5)

    _save(fig, "fig8")


# fig9 (ported from paper_build/v2/)
def fig9():
    import matplotlib.ticker as mticker

    # --- HARD values (manuscript) -- provenance paper_build/repro/phase6_robustness.out
    # Main curve: dead-zone (z=1.00, 5 GeV) max-region M SIC @ alpha=1e-3.
    CONTAM_FRAC_PCT = [0.0, 0.1, 1.0]            # calibration contamination fraction (%)
    CONTAM_SIC      = [30.0, 5.8, 0.94]          # PHASE 6.3 f=0/0.001/0.010 (manuscript HARD)

    # f=0 split-stability interval (PHASE 6.1, 20-split): median[min,max].
    F0_MED, F0_LO, F0_HI = 29.8, 28.0, 30.2

    # Coverage inset: achieved eps_B on a CLEAN held-out photon null at nominal
    # alpha=1e-3, across the FULL 6.3 ladder f={0, 0.1%, 1%, 5%}.
    NOMINAL_ALPHA = C.ALPHA if hasattr(C, "ALPHA") else 1e-3
    COV_FRAC_PCT  = [0.0, 0.1, 1.0, 5.0]
    COV_EPS_B     = [6.33e-4, 5.67e-4, 5.67e-4, 5.33e-4]  # PHASE 6.3 achieved eps_B(clean null)

    fig, (ax, axin) = plt.subplots(
        1, 2, figsize=(S.WIDTHS["double"], 3.3),
        gridspec_kw={"width_ratios": [2.05, 1.0], "wspace": 0.16},
        constrained_layout=True)
    x = np.arange(len(CONTAM_FRAC_PCT))

    # f=0 split-stability band drawn under the f=0 marker (the only f with a
    # committed seed/split interval; later f have a single split each).
    ax.errorbar([x[0]], [F0_MED],
                yerr=[[F0_MED - F0_LO], [F0_HI - F0_MED]],
                fmt="none", ecolor=S.CHANNELS["M"]["color"],
                elinewidth=1.4, capsize=4, capthick=1.4, zorder=4)

    # main power-decay curve in the frozen M (max-region) channel hue/marker.
    S.plot_channel(ax, x, CONTAM_SIC, "M", lw=1.6, ms=8, zorder=5, label=False)

    # numeric value labels to the RIGHT of each marker (consistent 2-decimal
    # precision; integers shown as-is). The f=0 marker sits at the top-left so
    # its bold value goes BELOW-right, clearing the ceiling line and the
    # interval annotation that lives in the open mid-upper space.
    val_offsets = [(11, -10), (11, 1), (11, 1)]        # f=0 dropped below marker
    for xi, v, off in zip(x, CONTAM_SIC, val_offsets):
        ax.annotate(f"{v:.2f}".rstrip("0").rstrip("."),
                    (xi, v), textcoords="offset points", xytext=off,
                    ha="left", va="center", fontsize=9.5, fontweight="bold",
                    color=S.CHANNELS["M"]["color"])
    # f=0 split-stability is carried by the error bar on the f=0 marker (above);
    # the numeric interval 29.8 [28.0, 30.2] and the deployment guide
    # (needs f << alpha = 0.1%) go in the caption, not on the axes.
    S.sic_ceiling(ax)                                  # 1/sqrt(alpha)=31.6 ref (label at right end)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f:g}%" for f in CONTAM_FRAC_PCT])
    S.no_minor_x(ax)                                   # kill categorical notch ticks
    ax.set_xlim(-0.35, len(x) - 0.12)
    ax.set_ylim(0.6, 60)
    ax.set_xlabel("calibration contamination fraction $f$")
    ax.set_ylabel(r"dead-zone (max-region $M$) SIC @ $\alpha=10^{-3}$")
    ax.set_title("Power decays under contamination")
    S.add_panel_label(ax, "a")
    S.despine(ax)
    S.grid_y(ax)

    # Panel (b): achieved FPR on a clean null vs contamination fraction (extended
    # to f=5%), with the budget alpha=1e-3 as a horizontal line. Every point stays
    # below budget at all f -> coverage stays valid even as panel (a) power collapses.
    fb = np.array(COV_FRAC_PCT)                        # [0, 0.1, 1, 5] %
    xb = np.arange(len(fb))                            # even categorical spacing

    axin.axhline(NOMINAL_ALPHA, color="0.45", ls=(0, (4, 2)), lw=1.0, zorder=1)
    axin.text(xb[-1], NOMINAL_ALPHA * 1.03, r"budget $\alpha=10^{-3}$",
              ha="right", va="bottom", fontsize=7.5, color="0.4")
    axin.plot(xb, COV_EPS_B, marker="o", ms=7, ls="-", lw=1.2, color="0.30",
              markeredgecolor="white", markeredgewidth=0.6, zorder=5)

    axin.set_xticks(xb)
    axin.set_xticklabels([f"{f:g}%" for f in fb])
    S.no_minor_x(axin)
    axin.set_xlim(-0.35, len(fb) - 0.65)
    axin.set_ylim(0, 1.2e-3)
    axin.yaxis.set_major_locator(mticker.FixedLocator([0, 5e-4, 1e-3]))
    axin.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: {0.0: "0", 5e-4: r"$5{\times}10^{-4}$",
                      1e-3: r"$10^{-3}$"}.get(round(v, 7), "")))
    axin.set_xlabel("contamination fraction $f$", fontsize=9)
    axin.set_ylabel(r"achieved $\varepsilon_B$ (clean null)", fontsize=9)
    axin.set_title("Coverage stays valid", fontsize=10.5)
    S.add_panel_label(axin, "b")
    S.despine(axin)
    S.grid_y(axin)

    _save(fig, "fig9")


# figS1 (ported from paper_build/v2/)
def figS1():
    from matplotlib.colors import to_rgb

    # Panel-(b) source: the occupancy baseline ladder (matched-eps_B, C0-split).
    LADDER_CSV = REPO / "scan_outputs" / "revision" / "multiseed" / "occupancy_baseline_ladder.csv"
    ACHIEV_CSV = REPO / "scan_outputs" / "achievability_table.csv"

    # Rung columns in the ladder CSV, in increasing methodological sophistication.
    # (display label, csv column, channel hue used for the bar)
    RUNGS = [
        (r"raw $[O_0\!=\!0]$",      "sic_a_O0cut",     "occupancy"),
        (r"$E_0/E_{\rm dep}$ CP",   "sic_b_E0frac_cp", "occupancy"),
        (r"3-var LR",               "sic_c_3var_logit", "occupancy"),
        (r"$T_{\rm occ}$",          "sic_d_Tocc",      "occupancy"),  # hero rung
        (r"conformal",              "sic_e_conformal", "p_comb"),
    ]
    HERO_COL = "sic_d_Tocc"

    def _ladder_z124():
        """Return the z=1.24, 5 GeV ladder row + its matched achieved eps_B."""
        df = pd.read_csv(LADDER_CSV)
        row = df[(df.z_m == 1.24) & (df.m_chi_GeV == 5.0)]
        assert len(row) == 1, f"expected one z=1.24/5GeV ladder row, got {len(row)}"
        return row.iloc[0], float(row.iloc[0]["eb_a"])

    def _eps_floor():
        """Discrete-score achievability floor eps_eff for T_occ (floor_hit=True)."""
        df = pd.read_csv(ACHIEV_CSV)
        occ = df[(df.channel == "T_occ") & (df.floor_hit == True)]  # noqa: E712
        floors = occ["p_floor"].unique()
        assert len(floors) == 1, f"non-unique T_occ floor: {floors}"
        return float(floors[0])

    def build():
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(S.COL_FULL, 3.1))

        # Panel (a): layer-0 back-face collapse
        df = D.load_e0frac_vs_z(False).sort_values("z")
        occ_color = S.CHANNELS["occupancy"]["color"]
        occ_marker = S.CHANNELS["occupancy"]["marker"]

        axA.axvspan(1.04, 1.08, color=occ_color, alpha=0.10,
                    label="layer-0 back face")
        axA.plot(df.z, df.e0_frac_median, color=occ_color, marker=occ_marker,
                 ms=5, lw=1.4, zorder=3)
        axA.set_yscale("log")
        axA.set_xlabel(r"displacement $z$ [m]")
        axA.set_ylabel(r"median $E_0/E_{\rm dep}$ [dimensionless]")
        axA.set_title("Layer-0 back-face collapse", fontsize=10.5)
        S.grid_y(axA)
        axA.grid(True, which="both", axis="y", alpha=0.18, lw=0.5)

        # Annotate the documented 0.026 -> 0.001 collapse across the back face.
        for zk, val in C.E0FRAC_BACKFACE.items():
            z = float(zk.split("=")[1])
            left = z < 1.06
            axA.annotate(f"{val:g}", (z, val), textcoords="offset points",
                         xytext=(-6, 8) if left else (8, 2),
                         fontsize=8, ha="right" if left else "left", zorder=4)
        axA.legend(loc="lower left", fontsize=7.5, frameon=False)
        S.despine(axA)
        S.add_panel_label(axA, "a")

        # Panel (b): matched-achieved-eps_B occupancy ladder
        row, eb = _ladder_z124()
        eps_eff = _eps_floor()

        labels = [lab for lab, _, _ in RUNGS]
        vals = [float(row[col]) for _, col, _ in RUNGS]
        colors = [S.CHANNELS[hue]["color"] for _, _, hue in RUNGS]
        xpos = np.arange(len(RUNGS))

        bars = axB.bar(xpos, vals, color=colors, width=0.66,
                       edgecolor="black", lw=0.5, zorder=3)
        # Highlight the T_occ hero rung with a heavier edge.
        hero_i = [c for _, c, _ in RUNGS].index(HERO_COL)
        bars[hero_i].set_edgecolor("black")
        bars[hero_i].set_linewidth(1.4)

        # Numeric SIC value INSIDE the top of each bar (committed precision, 2 dp) so
        # no label crosses the ceiling line -- the E0/Edep bar (30.35) sits right under
        # 1/sqrt(alpha)=31.6, so an above-bar label would collide with the dotted line.
        drop = max(vals) * 0.018
        for x, v, fc in zip(xpos, vals, colors):
            tc = "white" if S._luminance(to_rgb(fc)) < 0.5 else "black"
            axB.text(x, v - drop, f"{v:.2f}", ha="center", va="top", fontsize=7.5,
                     fontweight="bold", color=tc, zorder=4)

        # SIC ceiling reference (1/sqrt(alpha) = 31.6). Label parked at the LEFT END of
        # the dotted line, in the clear space above the short raw-cut bar -- away from
        # the tall E0/Edep bar so nothing collides with the line.
        ceil_v = S.sic_ceiling(axB, label=False)
        axB.text(0.015, ceil_v, r"$1/\sqrt{\alpha}=%.1f$" % ceil_v,
                 transform=axB.get_yaxis_transform(),
                 ha="left", va="bottom", fontsize=8, color="0.3")

        axB.set_xticks(xpos)
        axB.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        axB.set_ylabel(r"SIC $=\varepsilon_S/\sqrt{\varepsilon_B}$ [dimensionless]")
        # Headroom: T_occ (39.4) sits ABOVE the 1/sqrt(alpha)=31.6 ceiling -- that
        # over-ceiling excess is the C13 operating-point/discrete-floor effect.
        # Extra top margin so the tall-bar value label clears the axes edge.
        axB.set_ylim(0, max(vals) * 1.26)
        S.no_minor_x(axB)
        S.grid_y(axB)
        S.despine(axB)
        S.add_panel_label(axB, "b")
        axB.set_title(r"Occupancy ladder ($z=1.24$ m, 5 GeV)", fontsize=10.5)

        # (Operating point goes in the caption, not on the axes: matched
        # eps_B = 4.33e-4 -> 1/eps_B = 2309, discrete floor eps_eff = 1.25e-4.)
        return fig, dict(vals=dict(zip([c for _, c, _ in RUNGS], vals)),
                         eb=eb, eps_eff=eps_eff,
                         e0_low=float(df.e0_frac_median.min()))

    fig, info = build()
    _save(fig, "figS1")


# dispatch
FIGURES = {
    "fig_eventpanel": fig_eventpanel,   # Fig 1(b) event display
    "fig2": fig2,                       # Fig 2  calibration coverage
    "fig3": fig3,                       # Fig 3  score distributions
    "fig4": fig4,                       # Fig 4  best-channel SIC heatmap (seed 001)
    "fig5": fig5,                       # Fig 5  mechanism: mass dependence
    "fig6": fig6,                       # Fig 6  dead-zone recovery bars
    "fig7": fig7,                       # Fig 7  core-depletion / latent regions
    "fig8": fig8,                       # Fig 8  collinear blind spot
    "fig9": fig9,                       # Fig 9  contamination scan
    "figS1": figS1,                     # Fig 10 occupancy mechanism
}


def main():
    ap = argparse.ArgumentParser(description="Regenerate paper figures from committed inputs.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="build every figure (default)")
    g.add_argument("--fig", choices=list(FIGURES), help="build a single figure")
    args = ap.parse_args()
    if args.fig:
        FIGURES[args.fig]()
    else:
        for name, fn in FIGURES.items():
            fn()


if __name__ == "__main__":
    main()
