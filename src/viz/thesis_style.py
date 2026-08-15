"""Shared plotting style for all thesis figures.

Single source of truth for colours, labels, figure sizes and matplotlib
rcParams so that every figure across every analysis notebook reads as one
visual system. Import and call `apply_style()` at the top of each notebook,
then draw with the exported PALETTE / LABEL dicts.

    from src.viz.thesis_style import (
        apply_style, W_COLORS, DS_COLORS, T_COLORS, DS_LABELS, T_LABELS,
        BISTAB_COLORS, VOTE_COLORS, TASK_COLORS, REF_COLOR,
        CMAP_SEQ, CMAP_DIV, FIGSIZES,
    )
    apply_style()

Design rules:
- Colours are colourblind-safe (Okabe-Ito base).
- Each analytical role has its own fixed colour map. Roles that ever appear
  together (dataset vs topology) use disjoint hues so blue/red never mean two
  different things in the same or an adjacent figure.
- No in-figure suptitles: the LaTeX caption carries the description.
"""

import matplotlib.pyplot as plt

# --- Okabe-Ito base palette (colourblind-safe) --------------------------------
BLUE   = '#0072B2'
ORANGE = '#E69F00'
GREEN  = '#009E73'
VERM   = '#D55E00'
PURPLE = '#CC79A7'
SKY    = '#56B4E9'
YELLOW = '#F0E442'
GREY   = '#7F7F7F'

# --- Fixed role -> colour maps ------------------------------------------------
# Datasets (blue / vermillion)
DS_COLORS = {'gpqa': BLUE, 'hiddenbench': VERM}
DS_LABELS = {'gpqa': 'GPQA', 'hiddenbench': 'HiddenBench'}
# Compact abbreviation for tight axes/legends (single spelling everywhere).
ABBR = {'gpqa': 'GPQA', 'hiddenbench': 'HB'}

# Topology: green / purple, disjoint from the dataset pair to avoid the
# blue=fc-vs-blue=GPQA collision that made the old figures ambiguous.
T_COLORS = {'fc': GREEN, 'star': PURPLE}
T_LABELS = {'fc': 'Fully connected', 'star': 'Star'}

# Memory window is ordinal -> single-hue sequential blues (light -> dark).
W_COLORS = {1: '#9ECAE1', 2: '#3182BD', 5: '#08519C'}

# Bistability cell labels
BISTAB_COLORS = {'monostable': BLUE, 'multistable': VERM, 'stochastic': GREY}

# Vote options (only ever appear alone, in the limit-cycle timelines)
VOTE_COLORS = {'A': BLUE, 'B': ORANGE, 'C': GREEN, 'D': VERM}

# High-repetition tasks (only ever appear alone, in the high-rep section)
TASK_COLORS = {'q84': BLUE, 'q125': ORANGE, 'q144': GREEN}

# Reference / null / baseline lines
REF_COLOR = GREY

# Heatmap colormaps: sequential for unsigned magnitudes, diverging for signed.
CMAP_SEQ = 'cividis'
CMAP_DIV = 'RdBu_r'

# --- Standard figure sizes (inches) -------------------------------------------
# Consistent widths per layout class so on-page text size is uniform once the
# matching \includegraphics width is applied in LaTeX.
FIGSIZES = {
    'single':   (7.0, 4.3),    # one panel            -> 0.72\linewidth
    'twopanel': (12.0, 4.8),   # 1x2 panels           -> 0.85\linewidth
    'wide':     (13.0, 4.6),   # 1x2 wide / 1x3       -> \linewidth
    'grid1x3':  (14.0, 4.6),   # 1x3 panels           -> \linewidth
    'grid2x2':  (12.0, 9.2),   # 2x2 panels           -> 0.9\linewidth
    'grid2x3':  (15.0, 9.0),   # 2x3 panels           -> \linewidth
    'tall':     (7.5, 8.5),    # tall single (lollipop)-> 0.9\linewidth
}


def apply_style():
    """Set global matplotlib rcParams for all thesis figures."""
    plt.rcParams.update({
        'figure.dpi': 130,
        'savefig.dpi': 200,
        'savefig.bbox': 'tight',
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.color': '#DDDDDD',
        'grid.linestyle': '-',
        'grid.linewidth': 0.6,
        'grid.alpha': 0.8,
        'axes.axisbelow': True,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#CCCCCC',
        'text.usetex': False,
        'mathtext.default': 'regular',
    })


def no_grid(*axes):
    """Disable the grid on axes that hold images (heatmaps, recurrence, timelines)."""
    for ax in axes:
        ax.grid(False)
