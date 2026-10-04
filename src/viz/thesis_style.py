"""Shared thesis figure style (Okabe-Ito, fixed role colours, sizes). Call apply_style() first."""

import matplotlib.pyplot as plt

BLUE   = '#0072B2'
ORANGE = '#E69F00'
GREEN  = '#009E73'
VERM   = '#D55E00'
PURPLE = '#CC79A7'
SKY    = '#56B4E9'
YELLOW = '#F0E442'
GREY   = '#7F7F7F'

DS_COLORS = {'gpqa': BLUE, 'hiddenbench': VERM}
DS_LABELS = {'gpqa': 'GPQA', 'hiddenbench': 'HiddenBench'}
ABBR = {'gpqa': 'GPQA', 'hiddenbench': 'HB'}

T_COLORS = {'fc': GREEN, 'star': PURPLE}
T_LABELS = {'fc': 'Fully connected', 'star': 'Star'}

W_COLORS = {1: '#9ECAE1', 2: '#3182BD', 5: '#08519C'}

BISTAB_COLORS = {'monostable': BLUE, 'multistable': VERM, 'stochastic': GREY}

VOTE_COLORS = {'A': BLUE, 'B': ORANGE, 'C': GREEN, 'D': VERM}

TASK_COLORS = {'q84': BLUE, 'q125': ORANGE, 'q144': GREEN}

REF_COLOR = GREY

CMAP_SEQ = 'cividis'
CMAP_DIV = 'RdBu_r'

# Inches; trailing comment = matching LaTeX \includegraphics width.
FIGSIZES = {
    'single':   (7.0, 4.3),    # 0.72\linewidth
    'twopanel': (12.0, 4.8),   # 0.85\linewidth
    'wide':     (13.0, 4.6),   # \linewidth
    'grid1x3':  (14.0, 4.6),   # \linewidth
    'grid2x2':  (12.0, 9.2),   # 0.9\linewidth
    'grid2x3':  (15.0, 9.0),   # \linewidth
    'tall':     (7.5, 8.5),    # 0.9\linewidth
}


def apply_style():
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
    for ax in axes:
        ax.grid(False)
