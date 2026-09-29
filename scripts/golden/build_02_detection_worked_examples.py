import os
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def co(s): cells.append(nbf.v4.new_code_cell(s))

md(r"""# GOLDEN 02_detection_worked_examples

**Thesis section:** `sec:method:sr`, `sec:method:bistability`, `sec:method:dominance`, `sec:method:lc`, `sec:method:vocab` (Chapter 5, `thesis/chapters/05_detection.tex`)
**Provenance:** Hand-authored pedagogical worked examples. NO data files — every number is reproduced with a tiny self-contained computation that runs through the SAME detector code path in `src/metrics/*.py`.
**Data:** none (illustrative examples only).
**Figures exported to:** none (this notebook verifies numbers, produces no figures).
**Status:** reproducible from committed source (`src/metrics/`), independent of any dataset.

This is a GOLDEN notebook: it recomputes every illustrative number in Chapter 5's worked-example paragraphs and prints each in a labelled block for manual copy into the thesis. Where a value is a hand-authored pedagogical constant it is checked against the literal `.tex` value with `MATCH`/`MISMATCH`. See `notebooks/golden/INDEX.md` for the full section->notebook map.
""")

md(r"""# Chapter 5 worked examples — number traceability

Verifies, reusing `src/metrics/` detector code:

- **SR** (`sec:method:sr`, L246-255): OLS segment slopes `c=(5,6,6,8,9)->+1.0`, `(7,7,6,6,5)->-0.5`, flat `->0`.
- **Bistability** (`sec:method:bistability`, L370-386): inverse-Simpson `N_eff` for `28A/2B->~1.14`, `15/15->2.0`, `10/10/10->3.0`.
- **Dominance** (`sec:method:dominance`, L500-518): the staggered-convergence conversion table -> `sigma=(61/6,25/6,5/3,0)`, shares `p~(0.635,0.260,0.104,0)`, `D~=0.31`, hub `~64%`; lone-dissenter `D=1/9~=0.11`.
- **Limit cycles** (`sec:method:lc`, L618-647): `v=(A,B,A,B,A,B)` -> `DET=1.0`, `P_hat=2`, combinatorial null `p=2/C(6,3)=0.10`; full-length `L=16` -> `~2/C(16,8)~=2e-4`; add-one floor `1/(B+1)~=1e-3` at `B=1000`. End-state guards for `(A,B,A,B,B,B,B)` (fixed point) and near-constant `(A,B,B,B,B,B,B)` (`P_hat=1`, rejected).
- **System macrostate count** (`sec:method:lc:system` / `sec:method:vocab`, L686): `C(N+M-1, M-1) = 15` for `M=3`, `35` for `M=4`, at `N=4`.
""")

co(r"""import sys
sys.path.insert(0, '../..')

from math import comb, isclose
from itertools import permutations

import numpy as np

# reuse the EXACT detector code paths
from src.metrics.self_reinforcement import extract_runs, _ols_slope
from src.metrics.bistability import _n_eff
from src.metrics.dominance import score_dominance
from src.metrics.limit_cycles import (
    _recurrence_matrix, _det, _diagonalwise_rr, _dominant_period, _trailing_k, _shuffle_p,
)


def chk(label, computed, thesis_says, tol):
    ok = 'MATCH' if abs(float(computed) - float(thesis_says)) <= tol else 'MISMATCH'
    print(f'  {label}: computed={computed:.4g}  thesis_says={thesis_says:.4g}  {ok}')


def make_rep(rounds, gt='A', topology=None):
    # rounds: list of rounds, each a list of (vote, confidence) tuples per agent
    traj = [{'phase_b': [{'vote': v, 'confidence': c} for (v, c) in rnd]} for rnd in rounds]
    rep = {'trajectory': traj, 'ground_truth': gt}
    if topology is not None:
        rep['topology'] = topology
    return rep

print('setup ok')""")

co(r"""# --- sec:method:sr worked example (05_detection.tex L246-255) ---
# One vote-stable segment (single agent holding A), OLS slope of confidence on round index.
print('--- sec:method:sr : vote-stable-segment OLS slopes (L246-255) ---')

sr_cases = {
    'up   c=(5,6,6,8,9)': ([5, 6, 6, 8, 9], +1.0),
    'down c=(7,7,6,6,5)': ([7, 7, 6, 6, 5], -0.5),
    'flat c=(6,6,6,6,6)': ([6, 6, 6, 6, 6],  0.0),
}
for name, (confs, thesis_beta) in sr_cases.items():
    rep = make_rep([[('A', c)] for c in confs], gt='A')
    runs = extract_runs([rep])
    assert len(runs) == 1, f'expected one vote-stable segment, got {len(runs)}'
    beta = runs[0]['slope']
    # cross-check the bare OLS helper on the same data
    beta_ols = _ols_slope(np.arange(len(confs)), np.array(confs, dtype=float))
    assert isclose(beta, beta_ols)
    chk(f'beta_seg [{name}]', beta, thesis_beta, tol=1e-9)
""")

co(r"""# --- sec:method:bistability worked example (05_detection.tex L370-386) ---
# Coexistence reading: inverse-Simpson (Hill) N_eff over endpoint attractors.
print('--- sec:method:bistability : N_eff = 1/sum p^2 (inverse Simpson) (L370-386) ---')

bistab_cases = {
    'monostable 28A/2B': ({'A': 28, 'B': 2}, 1.14),
    'bistable   15A/15B': ({'A': 15, 'B': 15}, 2.0),
    'tristable  10/10/10': ({'A': 10, 'B': 10, 'C': 10}, 3.0),
    'stochastic 15A/15B': ({'A': 15, 'B': 15}, 2.0),
}
for name, (counter, thesis_neff) in bistab_cases.items():
    neff = _n_eff(counter)
    chk(f'N_eff [{name}]', neff, thesis_neff, tol=5e-3)
""")

co(r"""# --- sec:method:dominance worked example (05_detection.tex L500-518) ---
# fc topology, staggered convergence onto D. score_dominance runs the emergent
# influence graph + confidence-weighted 1/m credit split + normalised Herfindahl D.
print('--- sec:method:dominance : staggered-convergence hub (L500-518) ---')

FC = [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]

# vote table with confidence subscripts from the thesis:
#   t   ag0   ag1   ag2   ag3
#   0   D8    A6    B5    C5
#   1   D9    D6    B5    C5
#   2   D9    D7    D5    C5
#   3   D9    D7    D6    D5
dom_rounds = [
    [('D', 8), ('A', 6), ('B', 5), ('C', 5)],
    [('D', 9), ('D', 6), ('B', 5), ('C', 5)],
    [('D', 9), ('D', 7), ('D', 5), ('C', 5)],
    [('D', 9), ('D', 7), ('D', 6), ('D', 5)],
]
res = score_dominance(make_rep(dom_rounds, gt='A', topology=FC))
s, p, D, hub = res['s'], res['p'], res['D'], res['hub']

print(f'  sigma (out-strength) = {[round(x, 4) for x in s]}   thesis=[61/6, 25/6, 5/3, 0]={[round(v,4) for v in (61/6,25/6,5/3,0.0)]}')
chk('sigma_0 (=61/6)', s[0], 61 / 6, tol=1e-9)
chk('sigma_1 (=25/6)', s[1], 25 / 6, tol=1e-9)
chk('sigma_2 (=5/3) ', s[2], 5 / 3, tol=1e-9)
chk('sigma_3 (=0)   ', s[3], 0.0, tol=1e-9)
print(f'  shares p = {[round(x, 4) for x in p]}   thesis~(0.635, 0.260, 0.104, 0)')
chk('p_0', p[0], 0.635, tol=2e-3)
chk('p_1', p[1], 0.260, tol=2e-3)
chk('p_2', p[2], 0.104, tol=2e-3)
sum_p2 = float(np.dot(p, p))
chk('sum p^2', sum_p2, 0.482, tol=2e-3)
chk('D (normalised Herfindahl)', D, 0.31, tol=6e-3)
chk('hub influence share %%', 100 * p[hub], 64.0, tol=0.6)
print(f'  hub agent index = {hub} (thesis: agent 0)')
""")

co(r"""# --- sec:method:dominance worked example: lone dissenter (05_detection.tex L518) ---
# A lone dissenter folding into a strong majority (3 already on A) -> D = 1/9.
print('--- sec:method:dominance : lone-dissenter fold D = 1/9 (L518) ---')

lone_rounds = [
    [('A', 5), ('A', 5), ('A', 5), ('B', 5)],
    [('A', 5), ('A', 5), ('A', 5), ('A', 5)],
    [('A', 5), ('A', 5), ('A', 5), ('A', 5)],
    [('A', 5), ('A', 5), ('A', 5), ('A', 5)],
]
res_lone = score_dominance(make_rep(lone_rounds, gt='A', topology=FC))
print(f'  sigma = {[round(x, 4) for x in res_lone["s"]]}   shares p = {[round(x, 4) for x in res_lone["p"]]}')
chk('D lone-dissenter (=1/9)', res_lone['D'], 1 / 9, tol=1e-9)
""")

co(r"""# --- sec:method:lc agent-level worked example (05_detection.tex L618-647) ---
# Alternating v=(A,B,A,B,A,B): categorical RQA (radius 0, no embedding).
print('--- sec:method:lc : alternating cycle RQA, L=6 (L618-638) ---')

seq6 = list('ABABAB')
k6 = _trailing_k(seq6)
R6 = _recurrence_matrix(seq6)
det6 = _det(R6)
rr6 = _diagonalwise_rr(R6)          # rr6[0]=RR(1), rr6[1]=RR(2), rr6[2]=RR(3), ...
P6 = _dominant_period(rr6)

print(f'  trailing constant k = {k6}  (candidate needs k < u=3)  L = {len(seq6)} (>=4 ok)')
print(f'  recurrence matrix R (main diag excluded):')
print(R6.astype(int))
print(f'  RR(1)={rr6[0]:.3f}  RR(2)={rr6[1]:.3f}  RR(3)={rr6[2]:.3f}   thesis: 0, 1, 0')
chk('DET (=1.0)', det6, 1.0, tol=1e-9)
chk('P_hat (=2)', P6, 2, tol=1e-9)
print(f'  period guard: 2 <= P_hat({P6}) <= floor(L/2)={len(seq6)//2}  -> {2 <= P6 <= len(seq6)//2}')

# exact combinatorial null over the vote multiset {3A, 3B}: fraction of the C(6,3)
# distinct arrangements whose DET >= observed DET (only ABABAB / BABABA reach 1.0).
uniq6 = set(permutations(seq6))
n_ge = sum(1 for a in uniq6 if _det(_recurrence_matrix(list(a))) >= det6 - 1e-12)
p_comb6 = n_ge / len(uniq6)
print(f'  distinct arrangements C(6,3) = {len(uniq6)} (=comb(6,3)={comb(6,3)}); #DET>=obs = {n_ge}')
chk('combinatorial null p (=2/20=0.10)', p_comb6, 2 / comb(6, 3), tol=1e-9)
chk('  cross-check 2/C(6,3)', 2 / comb(6, 3), 0.10, tol=1e-9)

# add-one shuffle p-value (Theiler Algorithm 0) tracks the combinatorial value at L=6
p_shuffle6 = _shuffle_p(seq6, det6, B=1000, rng=np.random.default_rng(0))
print(f'  add-one shuffle p (B=1000, seed=0) = {p_shuffle6:.3f}  (~0.10; does NOT clear 0.05)')
""")

co(r"""# --- sec:method:lc : full-length cycle L=16 and add-one floor (05_detection.tex L639-644) ---
print('--- sec:method:lc : full-length L~16 exact null and add-one floor (L639-644) ---')

seq16 = list('AB' * 8)      # L = 16, perfect period-2
det16 = _det(_recurrence_matrix(seq16))
P16 = _dominant_period(_diagonalwise_rr(_recurrence_matrix(seq16)))
print(f'  L=16 alternating: DET={det16:.3f}  P_hat={P16}')
chk('DET L=16 (=1.0)', det16, 1.0, tol=1e-9)
chk('P_hat L=16 (=2)', P16, 2, tol=1e-9)

p_exact16 = 2 / comb(16, 8)
print(f'  exact combinatorial null p = 2/C(16,8) = 2/{comb(16,8)} = {p_exact16:.3e}')
chk('exact null p (~2e-4)', p_exact16, 2e-4, tol=1e-4)

B = 1000
floor = 1 / (B + 1)
print(f'  add-one p-value floor 1/(B+1) at B={B} = {floor:.3e}')
chk('add-one floor (~1e-3)', floor, 1e-3, tol=1e-4)
print('  interpretation: exact null (~2e-4) < shuffle floor (~1e-3), so a full-length')
print('  perfect cycle is flagged decisively at the 1/(B+1) floor.')
""")

co(r"""# --- sec:method:lc : end-state guards (05_detection.tex L644-647) ---
print('--- sec:method:lc : fixed-point exclusion and period guard (L644-647) ---')

# (A,B,A,B,B,B,B): trailing constant run k >= u=3 -> fixed point, excluded.
seq_fp = list('ABABBBB')
k_fp = _trailing_k(seq_fp)
print(f'  (A,B,A,B,B,B,B): trailing k = {k_fp}  -> fixed_point (k>=3)? {k_fp >= 3}')
chk('trailing k fixed-point (>=3)', float(k_fp >= 3), 1.0, tol=1e-9)

# (A,B,B,B,B,B,B): near-constant; high DET but dominant period P_hat = 1
# (persistence, not oscillation) -> rejected by the period guard 2 <= P_hat.
seq_pc = list('ABBBBBB')
R_pc = _recurrence_matrix(seq_pc)
det_pc = _det(R_pc)
P_pc = _dominant_period(_diagonalwise_rr(R_pc))
print(f'  (A,B,B,B,B,B,B): DET={det_pc:.3f} (high)  P_hat={P_pc}  -> period guard rejects (P_hat<2)? {P_pc < 2}')
chk('near-constant P_hat (=1)', P_pc, 1, tol=1e-9)
""")

co(r"""# --- sec:method:lc:system / sec:method:vocab : macrostate count (05_detection.tex L686) ---
# Number of vote compositions = C(N+M-1, M-1) (stars and bars) at N=4 agents.
print('--- sec:method:lc:system : macrostate count C(N+M-1, M-1) at N=4 (L686) ---')

N = 4
for M, thesis_count in ((3, 15), (4, 35)):
    count = comb(N + M - 1, M - 1)
    chk(f'#macrostates M={M} (=C({N+M-1},{M-1}))', count, thesis_count, tol=1e-9)
""")

nb['cells'] = cells
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
from pathlib import Path as _P
_out = _P(__file__).resolve().parents[2] / 'notebooks' / 'golden' / '02_detection_worked_examples.ipynb'
_out.parent.mkdir(parents=True, exist_ok=True)
import nbformat as _nbf
with open(_out, 'w') as _fh: _nbf.write(nb, _fh)
print('wrote', _out, 'with', len(cells), 'cells')
