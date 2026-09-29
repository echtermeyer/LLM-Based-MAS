import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.viz.thesis_style import apply_style, BLUE, VERM, GREY

apply_style()

fig, ax = plt.subplots(figsize=(5.2, 3.0))

x = np.linspace(-1.7, 1.7, 600)
V = (x**2 - 1)**2

ax.fill_between(x[x <= 0], V[x <= 0], 1.6, color=BLUE,  alpha=0.10, zorder=1)
ax.fill_between(x[x >= 0], V[x >= 0], 1.6, color=VERM, alpha=0.10, zorder=1)

ax.plot(x, V, color='#2b2b2b', lw=2.0, zorder=3)

ax.plot(-1, 0, 'o', ms=9, color=BLUE,  zorder=5)
ax.plot( 1, 0, 'o', ms=9, color=VERM, zorder=5)
ax.plot( 0, 1, 's', ms=7, color=GREY, zorder=5)

ax.axvline(0, color=GREY, lw=0.9, ls='--', alpha=0.6, zorder=2)

ax.text(-1.0, -0.12, 'Basin A', ha='center', va='top', fontsize=9, color=BLUE)
ax.text( 1.0, -0.12, 'Basin B', ha='center', va='top', fontsize=9, color=VERM)
ax.text( 0.08, 1.08, 'unstable\n(separatrix)', ha='left', va='bottom',
         fontsize=8, color=GREY)

ax.set_xlabel('State  $x$', fontsize=10)
ax.set_ylabel('Potential  $V(x)$', fontsize=10)
ax.set_xlim(-1.7, 1.7)
ax.set_ylim(-0.22, 1.6)
ax.set_xticks([-1, 0, 1])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '../thesis/plots/bg/double_well.pdf')
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'Saved: {out}')
