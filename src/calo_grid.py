"""Calorimeter event-display helpers (shared theme).

VOXEL ORDERING  (AD_with_CF / CaloGAN 504-vector)
-------------------------------------------------
A shower is a length-504 vector laid out as the row-major (C-order) concatenation of the
three layer images, in this order:

    voxels[0:288]   -> layer 0,  reshape (3, 96)    (eta x phi strip)
    voxels[288:432] -> layer 1,  reshape (12, 12)   (shower maximum)
    voxels[432:504] -> layer 2,  reshape (12, 6)

This matches how the HDF5 stores `layer_0 (N,3,96)`, `layer_1 (N,12,12)`, `layer_2 (N,12,6)`
(flatten each and concatenate). If a particular dataset turns out to use a different in-layer
convention, set `transpose=True` (swaps rows/cols per layer) when loading; it is OFF by
default and the assumption above is the documented one.

All drawing routines assume `style.apply_style()` has been called by the figure script, so
fonts and the magma colormap match the rest of the figure set.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Rectangle

import style as _S   # shared theme: overlay colors, image colormap

LAYER_SHAPES = {"layer0": (3, 96), "layer1": (12, 12), "layer2": (12, 6)}
LAYER_SLICES = {"layer0": (0, 288), "layer1": (288, 432), "layer2": (432, 504)}
LAYER_TITLES = {"layer0": "L0", "layer1": "L1", "layer2": "L2"}
CMAP_IMAGE = "viridis"  # match the parent paper's (Krause et al.) event-display heatmaps
CORE_HALF = 1.5   # |row-5.5|,|col-5.5| <= 1.5 -> central 4x4 = 16 voxels


def load_shower(source, transpose: bool = False) -> dict:
    """Return {'layer0','layer1','layer2', ...extras} from a 504-vector, a path to an
    .npz with a 'voxels' array, or a dict carrying 'voxels' (+ optional 'second_photon_xy',
    'argmax_region')."""
    extras = {}
    if isinstance(source, (str, Path)):
        d = np.load(source, allow_pickle=True)
        v = np.asarray(d["voxels"]); extras = {k: d[k] for k in d.files if k != "voxels"}
    elif isinstance(source, dict):
        v = np.asarray(source["voxels"])
        extras = {k: val for k, val in source.items() if k != "voxels"}
    else:
        v = np.asarray(source)
    v = v.reshape(-1)
    assert v.size == 504, f"expected 504 voxels, got {v.size}"
    out = {}
    for name, (a, b) in LAYER_SHAPES.items():
        s, e = LAYER_SLICES[name]
        img = v[s:e].reshape(a, b)
        out[name] = img.T if transpose else img
    out.update(extras)
    return out


def energy_norm(images, log_energy: bool = True, floor_frac: float = 1e-3):
    """Shared color normalization across a set of layer images (e.g. all six layers of a
    signal/photon comparison) so the energy scale is directly comparable."""
    arrs = [np.asarray(im) for im in images]
    vmax = max((float(im.max()) for im in arrs), default=1.0) or 1.0
    if not log_energy:
        return Normalize(0, vmax)
    pos = [im[im > 0].ravel() for im in arrs if (im > 0).any()]
    vmin = max(float(np.concatenate(pos).min()), vmax * floor_frac) if pos else vmax * floor_frac
    return LogNorm(vmin=vmin, vmax=vmax)


def draw_layers(axes, shower, fig=None, norm=None, log_energy=True, colorbar=True,
                cmap=CMAP_IMAGE, hide_ticks=True):
    """Render the three layers as heatmaps on `axes` (length-3) with an optional shared
    colorbar. Empty (zero-energy) voxels render as the colormap's 'bad' (white), matching
    the parent paper's event displays. Pass hide_ticks=False to show eta/phi Cell ID ticks."""
    imgs = [shower["layer0"], shower["layer1"], shower["layer2"]]
    if norm is None:
        norm = energy_norm(imgs, log_energy=log_energy)
    cm = mpl.colormaps[cmap].copy(); cm.set_bad("white")
    ims = []
    for ax, key, img in zip(axes, ["layer0", "layer1", "layer2"], imgs):
        shown = np.ma.masked_less_equal(img, 0) if log_energy else img
        ims.append(ax.imshow(shown, cmap=cm, norm=norm, aspect="auto", interpolation="nearest"))
        ax.set_title(LAYER_TITLES[key], fontsize=9)
        if hide_ticks:
            ax.set_xticks([]); ax.set_yticks([])
    if colorbar and fig is not None:
        cb = fig.colorbar(ims[-1], ax=list(axes), fraction=0.045, pad=0.02)
        cb.set_label("Energy (MeV)")
    return ims, norm


def outline_core(ax_layer1, color=None, lw=1.6, label=None):
    """Outline the 16-voxel layer-1 core (central 4x4) on a layer-1 axes."""
    color = color or _S.OVERLAY["core"]
    ax_layer1.add_patch(Rectangle((3.5, 3.5), 4, 4, fill=False, edgecolor=color, lw=lw,
                                  label=label))


def highlight_region(ax_layer1, region="core", color=None, alpha=0.28):
    """Shade the layer-1 region selected by the max-region statistic (default: the core)."""
    color = color or _S.OVERLAY["core"]
    if region in ("core", "L1 core", "L1core"):
        ax_layer1.add_patch(Rectangle((3.5, 3.5), 4, 4, facecolor=color, edgecolor="none",
                                      alpha=alpha, zorder=3))
    elif isinstance(region, np.ndarray) and region.shape == (12, 12):
        ax_layer1.imshow(np.ma.masked_where(~region.astype(bool), np.ones((12, 12))),
                         cmap=mpl.colors.ListedColormap([color]), alpha=alpha, aspect="auto",
                         zorder=3)


def mark_point(ax_layer1, xy, color=None, ms=11, label=None):
    """Mark a deposit location (e.g. the second-photon centroid) on layer 1. xy = (col, row)."""
    if xy is None:
        return
    color = color or _S.OVERLAY["point"]
    xy = np.asarray(xy).reshape(2)
    ax_layer1.plot(xy[0], xy[1], marker="x", ms=ms, mew=2.4, color=color, zorder=5,
                   label=label)
