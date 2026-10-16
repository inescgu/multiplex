#!/usr/bin/env python3
"""
painfig.py -- one place for the look of every figure that goes in the deck.

Import this at the top of each fig_*.py script and call `use_theme()` once.
Nothing in here touches your pipeline; it only changes how figures look, so
re-running a fig_*.py script never invalidates a result.

Why a shared module
-------------------
The deck has a palette (deep blue #004A99, mid blue #3E6BAA, orange #E05A2B,
pale blue cards #F1F5FB) and a type hierarchy. Matplotlib defaults fight that
on every slide: tab10 blue/orange is close enough to the deck's blue/orange to
look like a mistake rather than a choice, titles are set in a different weight
to the slide titles, and every script picks its own figure size so the same
axis label renders at four different physical sizes across four slides. The
deck's own QA rule -- "don't change how chart labels look from chart to chart"
-- is impossible to satisfy if each script styles itself.

So: one palette, one font stack, one set of canvas sizes keyed to the slots on
the slides, and one rule about titles (the SLIDE carries the title; the figure
does not repeat it). Figures are written transparent so they sit on the white
slide without a grey card behind them.

Canvas sizes
------------
Figures are sized in inches to the slot they occupy on a 10 x 5.625" slide, so
that text inside the figure comes out at the physical point size you asked for.
Do not resize the PNG in PowerPoint afterwards -- that is what makes axis text
drift between slides. If a figure needs to be bigger, change the SLOT here and
re-run.

Usage
-----
    from painfig import use_theme, C, SLOT, save, fiber_color, no_title_banner
    use_theme()
    fig, ax = plt.subplots(figsize=SLOT["half"])
    ...
    save(fig, out / "fig_whatever.png")
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ---------------------------------------------------------------------------
# Palette -- taken from the deck's own slide XML, not re-invented
# ---------------------------------------------------------------------------
class C:
    BLUE = "#004A99"        # titles, primary series, seeds
    BLUE_MID = "#3E6BAA"    # secondary series
    BLUE_PALE = "#9DB8D8"   # de-emphasised / "not significant"
    CARD = "#F1F5FB"        # pale card fill, also the lightest heatmap step
    ORANGE = "#E05A2B"      # the accent: significance, neighbors, "look here"
    ORANGE_LT = "#F2A65A"
    GREY = "#8A94A6"        # axes furniture, non-significant bars
    GREY_LT = "#D7DCE4"
    INK = "#17212E"         # body text inside figures
    MUTED = "#5B6675"       # captions, tick labels
    GREEN = "#2E7D5B"       # only for "control / reference" series
    RED = "#B3261E"         # only for a genuine warning (contradiction flags)


#: Sequential ramp for density / percentile heatmaps. Pale card -> deep blue.
CMAP_BLUE = LinearSegmentedColormap.from_list(
    "pain_blue", [C.CARD, "#C7D9EE", "#7CA3CE", "#2E6BAE", C.BLUE, "#002C5C"]
)
#: Sequential ramp for marker-strength heatmaps. Pale -> orange -> deep red.
CMAP_WARM = LinearSegmentedColormap.from_list(
    "pain_warm", ["#FBF2EC", "#F7D9BF", C.ORANGE_LT, C.ORANGE, "#A8381A"]
)
#: Diverging ramp for z-scores and AUC-0.5 (blue = below, orange = above).
CMAP_DIV = LinearSegmentedColormap.from_list(
    "pain_div", [C.BLUE, "#7CA3CE", "#FFFFFF", C.ORANGE_LT, C.ORANGE]
)

#: Fiber-type colours. Deliberately NOT the tab10 blue/orange/red/green the
#: pipeline currently uses -- those read as defaults next to the deck palette.
FIBER_COLORS = {
    "A-fiber": C.BLUE,
    "C-fiber": C.ORANGE,
    "ATF3": C.RED,
    "ATF3 (injured)": C.RED,
    "Non-neuronal": C.GREEN,
    "Non-classified": C.RED,
    "Unclassified": C.GREY_LT,
    "Global PPI (bridge)": C.GREY_LT,
    "unknown": C.GREY_LT,
}


def fiber_color(name: str) -> str:
    return FIBER_COLORS.get(name, C.GREY_LT)


# ---------------------------------------------------------------------------
# Canvas slots -- named after where the figure lands on the slide
# ---------------------------------------------------------------------------
SLOT = {
    "full": (9.2, 4.0),      # one figure, full content width
    "half": (4.5, 3.4),      # two side by side
    "half_tall": (4.5, 4.2),  # left column, title to footer
    "third": (2.95, 3.0),    # three across
    "band": (9.2, 2.0),      # wide and short, under a text block
    "strip": (4.5, 1.6),     # a single row of small multiples
}

#: Point sizes. These are the sizes the deck's own captions use, so the figure
#: text and the slide text match when the PNG is placed at 1:1.
FS = {"tick": 8, "label": 9, "annot": 8, "panel": 10, "big": 13}


def figsize(args_figsize, default):
    """--figsize W H overrides the default slot. Size the canvas to the slot the
    PNG will occupy on the slide: a figure generated at 9.2in and placed at 6in
    has all its text shrunk by a third, which is exactly how axis labels end up
    looking different from one slide to the next."""
    if not args_figsize:
        return default
    w, h = (float(v) for v in args_figsize)
    return (w, h)


def use_theme(base_font: str = "Arial") -> None:
    """Apply the deck look to every subsequent figure in this process."""
    plt.rcdefaults()
    plt.rcParams.update({
        # Arial is the deck's theme font; Liberation Sans is metric-compatible
        # and is what Linux boxes actually have, so the fallback is invisible.
        "font.family": "sans-serif",
        "font.sans-serif": [base_font, "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "font.size": FS["label"],
        "text.color": C.INK,
        "axes.labelcolor": C.MUTED,
        "axes.labelsize": FS["label"],
        "axes.titlesize": FS["panel"],
        "axes.titleweight": "bold",
        "axes.titlecolor": C.BLUE,
        "axes.titlelocation": "left",
        "axes.titlepad": 7,
        "axes.edgecolor": C.GREY_LT,
        "axes.linewidth": 0.8,
        "axes.facecolor": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.color": C.MUTED,
        "ytick.color": C.MUTED,
        "xtick.labelsize": FS["tick"],
        "ytick.labelsize": FS["tick"],
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
        "legend.fontsize": FS["tick"],
        "legend.handlelength": 1.2,
        "legend.handleheight": 0.8,
        "legend.labelspacing": 0.35,
        "figure.facecolor": "none",
        "figure.dpi": 110,
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "lines.linewidth": 1.4,
        "patch.linewidth": 0,
    })


# ---------------------------------------------------------------------------
# Small helpers used by more than one script
# ---------------------------------------------------------------------------
def panel_title(ax, text: str, sub: str | None = None) -> None:
    """Panel heading inside a multi-panel figure. The SLIDE carries the real
    title; this names the panel only, so it stays small and sentence-case."""
    ax.set_title(text, fontsize=FS["panel"], fontweight="bold",
                 color=C.BLUE, loc="left", pad=16 if sub else 7)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=FS["tick"],
                color=C.MUTED, va="bottom", ha="left")


def reference_line(ax, value: float, axis: str = "x", label: str | None = None) -> None:
    """The 'no effect' line (0.5 for AUC, 0 for a z-score). Always the same
    weight and colour so the eye learns it once."""
    fn = ax.axvline if axis == "x" else ax.axhline
    fn(value, color=C.MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    if label:
        (ax.text(value, 1.01, label, transform=ax.get_xaxis_transform(),
                 fontsize=FS["tick"], color=C.MUTED, ha="center", va="bottom")
         if axis == "x" else
         ax.text(1.01, value, label, transform=ax.get_yaxis_transform(),
                 fontsize=FS["tick"], color=C.MUTED, ha="left", va="center"))


def sig_colors(qvals, hit=None, miss=None):
    """Orange where BH q < 0.05, pale blue otherwise. One rule, every figure,
    so 'orange' always means the same thing across the whole deck."""
    import numpy as np
    hit = hit or C.ORANGE
    miss = miss or C.BLUE_PALE
    return [hit if (q is not None and not (isinstance(q, float) and np.isnan(q)) and q < 0.05)
            else miss for q in qvals]


def bar_value_labels(ax, bars, values, fmt="{:,.0f}", pad=0.01, color=None):
    """Value at the end of each horizontal bar, in the figure's own type size."""
    span = ax.get_xlim()[1] - ax.get_xlim()[0]
    for b, v in zip(bars, values):
        ax.text(b.get_width() + pad * span, b.get_y() + b.get_height() / 2,
                fmt.format(v), va="center", ha="left",
                fontsize=FS["annot"], color=color or C.INK)


def caption(fig, text: str, y: float = -0.01, chars_per_inch: float = 15.0) -> None:
    """One muted block under the figure for the method detail that would
    otherwise crowd the slide. Wrapped to the figure width so a long caption
    never runs off the PNG (savefig bbox='tight' would otherwise widen the
    whole image to fit one line, shrinking the plot on the slide)."""
    import textwrap
    width = max(40, int(fig.get_size_inches()[0] * chars_per_inch))
    fig.text(0.0, y, textwrap.fill(text, width), fontsize=FS["tick"],
             color=C.MUTED, ha="left", va="top", linespacing=1.45)


def save(fig, path, close: bool = True, dpi=None, formats=()) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)                      # dpi=None -> rcParams (300)
    for ext in formats:
        fig.savefig(path.with_suffix(f".{ext}"))
        print(f"  saved -> {path.with_suffix(f'.{ext}')}")
    if close:
        plt.close(fig)
    print(f"  saved -> {path}")
    return path


def shorten_layer(name: str, width: int = 22) -> str:
    """Cell-type names are long and the deck's figure slots are narrow.
    Shorten predictably rather than letting matplotlib clip."""
    s = str(name).replace("_", "/")
    return s if len(s) <= width else s[: width - 1] + "\u2026"