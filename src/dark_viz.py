"""
src/dark_viz.py
---------------
Shared dark-mode visualisation style for the volatility_regimes project.

All pipeline figures (Streamlit app + README) are built with this module so
they share a consistent dark aesthetic that matches Streamlit's default dark
theme (#0E1117 background).

Usage
-----
    from src.dark_viz import apply_style, C, savefig

    apply_style()                     # call once before any plt figure
    fig, ax = plt.subplots(...)
    ax.plot(x, y, color=C.HYBRID)
    savefig(fig, 'assets/figures/dark/my_chart.png')

Colour palette (C)
------------------
    C.BG         — figure / axes background   #0E1117
    C.GRID       — subtle grid lines           #1E2130
    C.TEXT       — all text / spines           #E0E0E0
    C.MUTED      — secondary text / ticks      #888888

    C.GARCH      — GARCH baseline              #4FC3F7  (light blue)
    C.EGARCH     — EGARCH baseline             #81D4FA  (lighter blue)
    C.HYBRID     — XGBoost hybrid              #FF7043  (deep orange)
    C.ACTUAL     — realised / y_true           #FAFAFA  (near-white)

    C.LOW        — HMM low-vol regime          #66BB6A  (green)
    C.MED        — HMM medium-vol regime       #FFA726  (amber)
    C.HIGH       — HMM high-vol regime         #EF5350  (red)

    C.UPPER      — upper bound / violation     #FF7043  (matches HYBRID)
    C.LOWER      — lower bound                 #4FC3F7  (matches GARCH)
    C.BAND       — interval fill (alpha 0.15)  #FF7043

    C.POSITIVE   — positive highlight          #66BB6A  (green)
    C.NEGATIVE   — negative / exception        #EF5350  (red)
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Palette:
    # Background / structural
    BG      : str = '#0E1117'
    GRID    : str = '#1E2130'
    TEXT    : str = '#E0E0E0'
    MUTED   : str = '#888888'

    # Model lines
    GARCH   : str = '#4FC3F7'
    EGARCH  : str = '#81D4FA'
    HYBRID  : str = '#FF7043'
    ACTUAL  : str = '#FAFAFA'

    # HMM regimes
    LOW     : str = '#66BB6A'
    MED     : str = '#FFA726'
    HIGH    : str = '#EF5350'

    # Conformal bands
    UPPER   : str = '#FF7043'
    LOWER   : str = '#4FC3F7'
    BAND    : str = '#FF7043'   # used with alpha=0.15 for fill_between

    # Generic highlights
    POSITIVE : str = '#66BB6A'
    NEGATIVE : str = '#EF5350'


C = _Palette()

# Map regime integer → colour for convenience
REGIME_COLOURS = {0: C.LOW, 1: C.MED, 2: C.HIGH}
REGIME_LABELS  = {0: 'Low', 1: 'Medium', 2: 'High'}


# ---------------------------------------------------------------------------
# Style application
# ---------------------------------------------------------------------------

def apply_style(font_scale: float = 1.0) -> None:
    """
    Apply the project-wide dark matplotlib style.

    Call once at the top of any figure-building script or notebook cell
    before creating any figure.  Safe to call multiple times.

    Parameters
    ----------
    font_scale : float
        Multiply all font sizes by this factor.  Use 1.2 for README-hero
        figures that will be viewed at smaller sizes.
    """
    base_fontsize = 11 * font_scale

    mpl.rcParams.update({
        # Canvas
        'figure.facecolor'      : C.BG,
        'axes.facecolor'        : C.BG,
        'savefig.facecolor'     : C.BG,
        'savefig.edgecolor'     : C.BG,

        # Text
        'text.color'            : C.TEXT,
        'axes.labelcolor'       : C.TEXT,
        'xtick.color'           : C.MUTED,
        'ytick.color'           : C.MUTED,
        'axes.titlecolor'       : C.TEXT,

        # Spines
        'axes.edgecolor'        : C.GRID,
        'axes.linewidth'        : 0.8,

        # Grid
        'axes.grid'             : True,
        'grid.color'            : C.GRID,
        'grid.linewidth'        : 0.6,
        'grid.alpha'            : 1.0,

        # Fonts
        'font.size'             : base_fontsize,
        'axes.titlesize'        : base_fontsize + 1,
        'axes.labelsize'        : base_fontsize,
        'xtick.labelsize'       : base_fontsize - 1,
        'ytick.labelsize'       : base_fontsize - 1,
        'legend.fontsize'       : base_fontsize - 1,

        # Legend
        'legend.framealpha'     : 0.25,
        'legend.facecolor'      : '#1A1D27',
        'legend.edgecolor'      : C.GRID,

        # Lines
        'lines.linewidth'       : 1.2,
        'patch.linewidth'       : 0.5,

        # Resolution
        'figure.dpi'            : 130,
        'savefig.dpi'           : 150,

        # Layout
        'figure.constrained_layout.use' : True,
    })


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def savefig(
    fig: plt.Figure,
    path: str | Path,
    *,
    tight: bool = True,
) -> Path:
    """
    Save *fig* to *path*, creating parent directories as needed.

    Parameters
    ----------
    fig   : matplotlib Figure
    path  : destination path (str or Path)
    tight : whether to call tight_layout before saving

    Returns
    -------
    Path  : resolved path of the saved file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass   # constrained_layout raises if called after tight_layout
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path.resolve()


# ---------------------------------------------------------------------------
# Shared figure factories (reused by both notebook and build_figures stage)
# ---------------------------------------------------------------------------

def regime_band_ax(
    ax: plt.Axes,
    regimes,          # pd.Series with DatetimeIndex, values in {0, 1, 2}
    alpha: float = 0.18,
) -> None:
    """
    Draw coloured background bands on *ax* for each HMM regime period.

    Parameters
    ----------
    ax      : target Axes
    regimes : Series of integer regime labels aligned to the same index
    alpha   : fill transparency
    """
    import pandas as pd
    regimes = pd.Series(regimes)
    prev_regime = None
    start_idx   = None

    for date, regime in regimes.items():
        if regime != prev_regime:
            if prev_regime is not None:
                ax.axvspan(start_idx, date,
                           color=REGIME_COLOURS[prev_regime], alpha=alpha,
                           linewidth=0)
            start_idx   = date
            prev_regime = regime

    # close the final band
    if prev_regime is not None and start_idx is not None:
        ax.axvspan(start_idx, regimes.index[-1],
                   color=REGIME_COLOURS[prev_regime], alpha=alpha,
                   linewidth=0)


def add_regime_legend(ax: plt.Axes) -> None:
    """Add a compact Low / Medium / High regime patch legend to *ax*."""
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color=C.LOW,  label='Low vol',    alpha=0.7),
        mpatches.Patch(color=C.MED,  label='Medium vol', alpha=0.7),
        mpatches.Patch(color=C.HIGH, label='High vol',   alpha=0.7),
    ]
    ax.legend(handles=patches, loc='upper left', framealpha=0.3)
