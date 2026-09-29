import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from src.viz.thesis_style import apply_style, BLUE, VERM, GREY

apply_style()

INK = '#2b2b2b'
fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 4.0),
                               gridspec_kw={'wspace': 0.18})

# ============================================================
# (a) Structural motif: coherent feed-forward loop, AND logic
# ============================================================
axL.set_xlim(-0.45, 2.45)
axL.set_ylim(-0.95, 1.85)
axL.set_aspect('equal')
axL.axis('off')

pos = {'A': (0.0, 1.35), 'B': (2.0, 1.35), 'C': (1.0, 0.0)}
R = 0.24
node_col = {'A': BLUE, 'B': BLUE, 'C': VERM}
for name, (x, y) in pos.items():
    axL.add_patch(Circle((x, y), R, facecolor=node_col[name],
                         edgecolor=INK, lw=1.6, zorder=4))
    axL.text(x, y, name, ha='center', va='center', color='white',
             fontsize=15, fontweight='bold', zorder=5)

for p, q in [('A', 'B'), ('A', 'C'), ('B', 'C')]:
    (x0, y0), (x1, y1) = pos[p], pos[q]
    d = np.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / d, (y1 - y0) / d
    axL.add_patch(FancyArrowPatch((x0 + ux * R, y0 + uy * R),
                                  (x1 - ux * R, y1 - uy * R),
                                  arrowstyle='-|>', mutation_scale=17,
                                  lw=1.9, color=INK, zorder=2,
                                  shrinkA=0, shrinkB=0))

# AND-logic tag on C's two inputs
axL.add_patch(FancyBboxPatch((0.80, 0.38), 0.40, 0.26,
              boxstyle='round,pad=0.02,rounding_size=0.08',
              facecolor='white', edgecolor=VERM, lw=1.3, zorder=3))
axL.text(1.0, 0.51, 'AND', ha='center', va='center', fontsize=10.5,
         color=VERM, fontweight='bold', zorder=4)

axL.text(0.02, 0.98, '(a)', transform=axL.transAxes, ha='left', va='top',
         fontsize=12, fontweight='bold', color=INK)
axL.set_title('Structural motif', fontsize=12, pad=6)
axL.text(0.5, -0.11, 'a recurring wiring pattern:\nwho influences whom',
         transform=axL.transAxes, ha='center', va='top', fontsize=9.5,
         color=GREY, style='italic')

# ============================================================
# (b) Dynamical motif: phase portrait with a stable limit cycle
# ============================================================
# normal form with a stable limit cycle at r = 1:
#   dx/dt = x - y - x(x^2+y^2),   dy/dt = x + y - y(x^2+y^2)
lim = 1.65
Y, X = np.mgrid[-lim:lim:300j, -lim:lim:300j]
R2 = X**2 + Y**2
U = X - Y - X * R2
V = X + Y - Y * R2
axR.streamplot(X, Y, U, V, color='#B9C2CC', density=1.15, linewidth=0.7,
               arrowsize=0.8, zorder=1)

# the limit cycle itself
th = np.linspace(0, 2 * np.pi, 400)
axR.plot(np.cos(th), np.sin(th), color=BLUE, lw=2.8, zorder=4)


def rk4(p0, n=1400, dt=0.01):
    def f(s):
        x, y = s
        r2 = x * x + y * y
        return np.array([x - y - x * r2, x + y - y * r2])
    out = [np.array(p0, float)]
    for _ in range(n):
        s = out[-1]
        k1 = f(s); k2 = f(s + dt / 2 * k1)
        k3 = f(s + dt / 2 * k2); k4 = f(s + dt * k3)
        out.append(s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
    return np.array(out)

traj = rk4([0.06, 0.02])                      # spirals out onto the cycle
axR.plot(traj[:, 0], traj[:, 1], color=VERM, lw=1.7, zorder=3)
axR.add_patch(FancyArrowPatch(traj[600], traj[615], arrowstyle='-|>',
              mutation_scale=15, lw=1.7, color=VERM, zorder=3))
axR.plot(0, 0, 'o', ms=6, mfc='white', mec=INK, mew=1.4, zorder=5)  # unstable fp

axR.set_xlim(-lim, lim)
axR.set_ylim(-lim, lim)
axR.set_aspect('equal')
axR.grid(False)
axR.set_xticks([])
axR.set_yticks([])
for sp in axR.spines.values():
    sp.set_visible(True)
    sp.set_color('#CCCCCC')
axR.text(0.02, 0.98, '(b)', transform=axR.transAxes, ha='left', va='top',
         fontsize=12, fontweight='bold', color=INK)
axR.set_title('Dynamical motif', fontsize=12, pad=6)
axR.text(0.5, -0.11, 'a recurring pattern in the trajectory:\nhow the system behaves over time',
         transform=axR.transAxes, ha='center', va='top', fontsize=9.5,
         color=GREY, style='italic')

out = os.path.join(os.path.dirname(__file__), '../thesis/plots/bg/motifs.pdf')
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'Saved: {out}')
