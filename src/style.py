"""Shared publication theme for the calorimeter-AD paper figures.

Every figure script imports this module and calls ``apply_style()`` once, so the
six figures share identical typography, palette, panel-label style, and save
format. Nothing here loads data; it is pure presentation.

Public API
----------
apply_style(use_tex=False)      set global rcParams (call once per process)
PALETTE                         fixed color code, learned once by the reader
LABELS                          identical legend strings everywhere
WIDTHS                          column widths in inches
add_panel_label(ax, "a")        bold lowercase "(a)" at axes-fraction (-0.02, 1.04)
despine(ax)                     drop top/right spines (line/bar/scatter axes)
grid_y(ax)                      light dotted y-grid (bar/line only)
add_demo_watermark(fig)         conspicuous diagonal "DEMO DATA" overlay
save_fig(fig, name)             write figures/out/<name>.{pdf,png}
DATA_IN, FIG_OUT                resolved input/output dirs (CWD-independent)
"""
from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Paths (resolved from this file, so scripts work from any CWD)
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_IN = REPO_ROOT / "data" / "in"
FIG_OUT = REPO_ROOT / "figures" / "out"

# ----------------------------------------------------------------------------
# Column widths (inches)
# ----------------------------------------------------------------------------
WIDTHS = {"single": 3.375, "onehalf": 5.0, "double": 6.9}

# Canonical figure sizes (inches). ONE width for the whole set so that, at the fixed
# rcParams font sizes, text and markers look identical across every figure; heights vary
# only by how many panel-rows a figure has.
FIG_W = 7.2
SIZES = {
    "wide12": (FIG_W, 3.0),   # one row of two panels (fig3, fig6, figS1)
    "single": (FIG_W, 4.0),   # single-panel figure (fig4, figS2)
    "events": (FIG_W, 4.3),   # two rows x three layer images (fig2c)
    "tall":   (FIG_W, 6.0),   # image row + a row of two panels (fig5)
}

# Overlay annotation colors for event displays (bright, readable on dark magma).
OVERLAY = {"core": "#19d3ff", "point": "#ff2bd1"}  # point=magenta (non-channel; avoids occupancy-green collision)

# ----------------------------------------------------------------------------
# Fixed color code  (identical across ALL figures)
# ----------------------------------------------------------------------------
PALETTE = {
    "T_raw": "#8c8c8c",       # scalar baseline (gray)
    "global": "#4e79a7",      # global ||z||^2 (steel blue)
    "M": "#c1432e",           # max-region HERO (vermillion)
    "supervised": "#2c3e50",  # supervised ceiling (navy)
    "photon": "#b0b0b0",      # photon-null control (light gray)
    "occupancy": "#59a14f",   # occupancy channel (green)
    "T_I": "#e69f00",         # Flow-I aggregate typicality (Okabe-Ito orange)
    "T_S": "#cc79a7",         # Flow-II morphology scalar (Okabe-Ito reddish-purple)
    "p_comb": "#6a3d9a",      # nested-conformal combined (purple)
}

# Named colormaps (spec-fixed)
CMAP_SIC = "viridis"     # sequential SIC magnitude (colorful, matches the original folder + parent paper)
CMAP_DIFF = "RdBu_r"     # diverging signal-photon difference (symmetric about 0)
CMAP_IMAGE = "magma"     # average shower images (energy magnitude)

# ----------------------------------------------------------------------------
# Identical legend strings everywhere
# ----------------------------------------------------------------------------
LABELS = {
    "T_raw": r"$T_{\rm raw}$",
    "global": r"global $\|z\|^2$",
    "M": r"$M$ (max-region)",
    "supervised": "supervised ceiling",
    "photon": "photon null",
    "occupancy": "occupancy channel",
    "T_I": r"$T_{\rm I}$",
    "T_S": r"$T_{\rm S}$",
    "T_occ": r"$T_{\rm occ}$",
    "p_comb": r"$p_{\rm comb}$",
}


def apply_style(use_tex: bool = False) -> None:
    """Set global rcParams. Call once per process. ``use_tex`` must be False
    for headless runs (no LaTeX installed); True is provided for final camera
    copy on a machine with a TeX toolchain."""
    mpl.rcParams.update({
        # vector output with embedded (Type-42) fonts
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "figure.constrained_layout.use": True,
        # serif, LaTeX-like typography
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 9.5,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        # ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        # legend
        "legend.frameon": False,
        # grid defaults (axes opt in via grid_y)
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
    })
    if use_tex:
        mpl.rcParams.update({
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}",
        })


def add_panel_label(ax, letter: str, x: float = -0.02, y: float = 1.04) -> None:
    """Bold lowercase ``(a)`` panel label at axes-fraction (x, y)."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="right")


def despine(ax) -> None:
    """Drop top/right spines (line/bar/scatter). Heatmaps/images keep all four."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def grid_y(ax) -> None:
    """Light dotted y-grid (bar/line panels only)."""
    ax.grid(axis="y", which="major", alpha=0.25, linestyle=":", linewidth=0.6)
    ax.set_axisbelow(True)


def add_demo_watermark(fig) -> None:
    """Conspicuous diagonal watermark for layout-test (--demo) figures."""
    fig.text(0.5, 0.5, "DEMO DATA", fontsize=26, color="#d62728", alpha=0.08,
             rotation=30, ha="center", va="center", zorder=1000, fontweight="bold")


def save_fig(fig, name: str) -> None:
    """Write ``figures/out/<name>.pdf`` and ``.png`` (300 dpi)."""
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    pdf = FIG_OUT / f"{name}.pdf"
    png = FIG_OUT / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    print(f"[saved] {pdf.relative_to(REPO_ROOT)}  +  {png.relative_to(REPO_ROOT)}")


# ============================================================================
# FROZEN SEMANTIC CHANNEL ENCODING  (Rule 2: ONE color + marker per channel,
# learned once and reused in every figure). Colorblind-safe (Okabe-Ito) AND
# grayscale-legible (distinct markers; linestyle carries the aggregate-vs-
# morphology dichotomy). Color comes from PALETTE, label from LABELS -- this
# block only adds the marker / linestyle / physics-family layer.
# ============================================================================

# REVTeX column widths (inches), named (single column / full width figure*).
COL_SINGLE = 3.375
COL_FULL = 7.0

# family: 'aggregate' (depends only on per-layer totals -> mass-independent)
#         'morphology' (intra-layer image shape -> collapses with opening angle)
#         'scalar' (raw mixed LL) | 'combined' | 'reference' | 'control'
_CH_SPEC = {
    "T_raw":      ("T_raw",      "o", "--", "scalar"),
    "T_I":        ("T_I",        "s", "-",  "aggregate"),
    "T_S":        ("T_S",        "D", "--", "morphology"),
    "M":          ("M",          "^", "-",  "combined"),
    "global":     ("global",     "v", "-",  "aggregate"),
    "T_occ":      ("occupancy",  "P", "-",  "aggregate"),
    "occupancy":  ("occupancy",  "P", "-",  "aggregate"),
    "p_comb":     ("p_comb",     "X", "-.", "combined"),
    "supervised": ("supervised", "",  ":",  "reference"),
    "photon":     ("photon",     ".", "-",  "control"),
}
CHANNELS = {
    name: {"color": PALETTE[ckey], "marker": mk, "ls": ls, "family": fam,
           "label": LABELS.get(name, name)}
    for name, (ckey, mk, ls, fam) in _CH_SPEC.items()
}

# family -> linestyle, for the aggregate-vs-morphology dichotomy (F5 line plot)
FAMILY_LS = {"aggregate": "-", "morphology": "--", "scalar": "--",
             "combined": "-", "reference": ":", "control": "-"}


def ch(name: str) -> dict:
    """Frozen style dict {color, marker, ls, family, label} for a channel."""
    return CHANNELS[name]


def plot_channel(ax, x, y, name, *, band=None, use_family_ls=False,
                 label=True, **kw):
    """Plot one channel in its frozen color + marker. ``band=(lo,hi)`` shades a
    min-max envelope in the same hue. ``use_family_ls`` selects the linestyle
    from the channel's physics family (for the aggregate/morphology figure)."""
    m = CHANNELS[name]
    ls = FAMILY_LS[m["family"]] if use_family_ls else kw.pop("ls", m["ls"])
    if band is not None:
        ax.fill_between(x, band[0], band[1], color=m["color"], alpha=0.18,
                        lw=0, zorder=1)
    lbl = m["label"] if label is True else (label or None)
    return ax.plot(x, y, color=m["color"], marker=m["marker"], ls=ls,
                   label=lbl, **kw)


def seed_band(ax, x, med, lo, hi, name, **kw):
    """Three-seed median[min,max]: shaded envelope + median line, channel hue."""
    return plot_channel(ax, x, med, name, band=(lo, hi), **kw)


def identity_diagonal(ax, lo, hi, **kw):
    """y = x reference for calibration (achieved-vs-nominal) panels."""
    opt = {"color": "0.4", "ls": "-", "lw": 0.9, "zorder": 0}
    opt.update(kw)
    ax.plot([lo, hi], [lo, hi], **opt)


def sic_ceiling(ax, value=None, *, label=True):
    """Horizontal 1/sqrt(alpha) SIC-saturation reference line."""
    import constants as _C
    v = _C.SIC_SATURATION if value is None else value
    ax.axhline(v, color="black", ls=":", lw=1.0, zorder=0)
    if label:
        ax.text(0.99, v, r"$1/\sqrt{\alpha}=%.1f$" % v,
                transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=8)
    return v


def floor_line(ax, eps, *, orient="v", label=None):
    """Achievability floor (eps_eff or per-bin 1/(n+1)) as a guide line."""
    (ax.axvline if orient == "v" else ax.axhline)(
        eps, color="0.55", ls=(0, (1, 1)), lw=0.9, zorder=0)
    if label:
        if orient == "v":
            ax.text(eps, 0.02, label, transform=ax.get_xaxis_transform(),
                    rotation=90, ha="right", va="bottom", fontsize=7, color="0.4")
        else:
            ax.text(0.02, eps, label, transform=ax.get_yaxis_transform(),
                    ha="left", va="bottom", fontsize=7, color="0.4")


def _luminance(rgba) -> float:
    return 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]


def heatmap_cell_text(ax, x, y, text, facecolor, **kw):
    """Print a value centered in a heatmap cell; auto black/white for contrast."""
    opt = {"ha": "center", "va": "center", "fontsize": 7}
    opt.update(kw)
    opt["color"] = "white" if _luminance(facecolor) < 0.55 else "black"
    ax.text(x, y, text, **opt)


def no_minor_x(ax):
    """Kill major+minor x ticks on categorical axes (prevents the global
    xtick.minor.visible 'notch' marks under category labels)."""
    ax.tick_params(axis="x", which="both", length=0)


def sic_rejection_axes(ax):
    """Standard SIC-vs-background-rejection axes: log x (1/eps_B), linear y."""
    ax.set_xscale("log")
    ax.set_xlabel(r"background rejection $1/\varepsilon_B$")
    ax.set_ylabel(r"SIC $=\varepsilon_S/\sqrt{\varepsilon_B}$")
