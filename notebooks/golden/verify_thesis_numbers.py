from pathlib import Path
import json
import sys

OUTPUTS = Path(__file__).parent / '_outputs'

def _load(name):
    p = OUTPUTS / f'{name}.json'
    if not p.exists():
        return None
    return json.loads(p.read_text())['numbers']

def _get(d, path):
    if d is None:
        return None
    val = d
    for part in path.split('.'):
        if not isinstance(val, dict):
            return None
        val = val.get(part)
    return val

CLAIMS = []

def claim(section, label, nb, path, expected, tol=0.001, note=None):
    CLAIMS.append((section, label, nb, path, expected, tol, note))

def exact(section, label, nb, path, expected, note=None):
    CLAIMS.append((section, label, nb, path, expected, 0, note))


# ─── nb03: System Performance ────────────────────────────────────────────────

exact('nb03 sec:results:perf', 'total_debates',           'nb03', 'total_debates',       30000)
exact('nb03 sec:results:perf', 'n_tasks_gpqa',            'nb03', 'n_tasks_gpqa',          50)
exact('nb03 sec:results:perf', 'n_tasks_hb',              'nb03', 'n_tasks_hb',             50)
exact('nb03 sec:results:perf', 'n_configs',               'nb03', 'n_configs',              12)

claim('nb03 sec:results:perf', 'gpqa_acc_per_agent',      'nb03', 'gpqa_acc_per_agent',   0.382)
claim('nb03 sec:results:perf', 'gpqa_acc_majority_r0',    'nb03', 'gpqa_acc_majority_r0', 0.405)
claim('nb03 sec:results:perf', 'gpqa_acc_after_debate',   'nb03', 'gpqa_acc_after_debate', 0.422)
claim('nb03 sec:results:perf', 'gpqa_acc_oracle_r0',      'nb03', 'gpqa_acc_oracle_r0',   0.706)
claim('nb03 sec:results:perf', 'hb_acc_per_agent',        'nb03', 'hb_acc_per_agent',     0.149)
claim('nb03 sec:results:perf', 'hb_acc_majority_r0',      'nb03', 'hb_acc_majority_r0',   0.076)
claim('nb03 sec:results:perf', 'hb_acc_after_debate',     'nb03', 'hb_acc_after_debate',  0.261)
claim('nb03 sec:results:perf', 'hb_acc_oracle_r0',        'nb03', 'hb_acc_oracle_r0',     0.463)

claim('nb03 app:perf:configs', 'spearman_rho_gpqa',       'nb03', 'spearman_rho_gpqa',   -0.54, tol=0.005)
claim('nb03 app:perf:configs', 'spearman_rho_hb',         'nb03', 'spearman_rho_hb',      0.08, tol=0.005)

claim('nb03 app:perf:configs', 'gp_fc_w1_acc',   'nb03', 'gp_fc_w1_acc',   0.414)
claim('nb03 app:perf:configs', 'gp_fc_w2_acc',   'nb03', 'gp_fc_w2_acc',   0.431)
claim('nb03 app:perf:configs', 'gp_fc_w5_acc',   'nb03', 'gp_fc_w5_acc',   0.424)
claim('nb03 app:perf:configs', 'gp_star_w1_acc', 'nb03', 'gp_star_w1_acc', 0.418)
claim('nb03 app:perf:configs', 'gp_star_w2_acc', 'nb03', 'gp_star_w2_acc', 0.425)
claim('nb03 app:perf:configs', 'gp_star_w5_acc', 'nb03', 'gp_star_w5_acc', 0.422)
claim('nb03 app:perf:configs', 'hi_fc_w1_acc',   'nb03', 'hi_fc_w1_acc',   0.252)
claim('nb03 app:perf:configs', 'hi_fc_w2_acc',   'nb03', 'hi_fc_w2_acc',   0.282)
claim('nb03 app:perf:configs', 'hi_fc_w5_acc',   'nb03', 'hi_fc_w5_acc',   0.258)
claim('nb03 app:perf:configs', 'hi_star_w1_acc', 'nb03', 'hi_star_w1_acc', 0.245)
claim('nb03 app:perf:configs', 'hi_star_w2_acc', 'nb03', 'hi_star_w2_acc', 0.266)
claim('nb03 app:perf:configs', 'hi_star_w5_acc', 'nb03', 'hi_star_w5_acc', 0.264)

claim('nb03 app:perf:configs', 'gp_fc_w1_rounds',   'nb03', 'gp_fc_w1_rounds',   3.87, tol=0.005)
claim('nb03 app:perf:configs', 'gp_fc_w2_rounds',   'nb03', 'gp_fc_w2_rounds',   3.68, tol=0.005)
claim('nb03 app:perf:configs', 'gp_fc_w5_rounds',   'nb03', 'gp_fc_w5_rounds',   3.83, tol=0.005)
claim('nb03 app:perf:configs', 'gp_star_w1_rounds', 'nb03', 'gp_star_w1_rounds', 4.29, tol=0.005)
claim('nb03 app:perf:configs', 'gp_star_w2_rounds', 'nb03', 'gp_star_w2_rounds', 4.23, tol=0.005)
claim('nb03 app:perf:configs', 'gp_star_w5_rounds', 'nb03', 'gp_star_w5_rounds', 4.36, tol=0.005)
claim('nb03 app:perf:configs', 'hi_fc_w1_rounds',   'nb03', 'hi_fc_w1_rounds',   5.18, tol=0.005)
claim('nb03 app:perf:configs', 'hi_fc_w2_rounds',   'nb03', 'hi_fc_w2_rounds',   4.71, tol=0.005)
claim('nb03 app:perf:configs', 'hi_fc_w5_rounds',   'nb03', 'hi_fc_w5_rounds',   4.79, tol=0.005)
claim('nb03 app:perf:configs', 'hi_star_w1_rounds', 'nb03', 'hi_star_w1_rounds', 7.13, tol=0.005)
claim('nb03 app:perf:configs', 'hi_star_w2_rounds', 'nb03', 'hi_star_w2_rounds', 6.48, tol=0.005)
claim('nb03 app:perf:configs', 'hi_star_w5_rounds', 'nb03', 'hi_star_w5_rounds', 6.27, tol=0.005)

claim('nb03 app:perf:configs', 'gp_fc_w1_p_cap%',   'nb03', 'gp_fc_w1_p_cap',   0.002)
claim('nb03 app:perf:configs', 'gp_fc_w2_p_cap%',   'nb03', 'gp_fc_w2_p_cap',   0.002)
claim('nb03 app:perf:configs', 'gp_fc_w5_p_cap%',   'nb03', 'gp_fc_w5_p_cap',   0.004)
claim('nb03 app:perf:configs', 'gp_star_w1_p_cap%', 'nb03', 'gp_star_w1_p_cap', 0.010)
claim('nb03 app:perf:configs', 'gp_star_w2_p_cap%', 'nb03', 'gp_star_w2_p_cap', 0.010)
claim('nb03 app:perf:configs', 'gp_star_w5_p_cap%', 'nb03', 'gp_star_w5_p_cap', 0.016)
claim('nb03 app:perf:configs', 'hi_fc_w1_p_cap%',   'nb03', 'hi_fc_w1_p_cap',   0.025)
claim('nb03 app:perf:configs', 'hi_fc_w2_p_cap%',   'nb03', 'hi_fc_w2_p_cap',   0.010)
claim('nb03 app:perf:configs', 'hi_fc_w5_p_cap%',   'nb03', 'hi_fc_w5_p_cap',   0.020)
claim('nb03 app:perf:configs', 'hi_star_w1_p_cap%', 'nb03', 'hi_star_w1_p_cap', 0.133)
claim('nb03 app:perf:configs', 'hi_star_w2_p_cap%', 'nb03', 'hi_star_w2_p_cap', 0.112)
claim('nb03 app:perf:configs', 'hi_star_w5_p_cap%', 'nb03', 'hi_star_w5_p_cap', 0.107)


# ─── nb04: Self-Reinforcement ─────────────────────────────────────────────────

exact('nb04 sec:results:sr',   'n_segments',    'nb04', 'n_segments',  133696)
claim('nb04 sec:results:sr',   'p_sr_min',      'nb04', 'p_sr_min',     0.733)
claim('nb04 sec:results:sr',   'p_sr_max',      'nb04', 'p_sr_max',     0.883)
claim('nb04 sec:results:sr',   'sr_eff_rho_min', 'nb04', 'sr_eff_rho_min', -0.82, tol=0.005)
claim('nb04 sec:results:sr',   'sr_eff_rho_max', 'nb04', 'sr_eff_rho_max', -0.73, tol=0.005)

claim('nb04 app:sr:supp', 'slope_min',          'nb04', 'slope_min',    0.354)
claim('nb04 app:sr:supp', 'slope_max',          'nb04', 'slope_max',    0.515)
claim('nb04 app:sr:supp', 'gp_fc_psr_min',      'nb04', 'gp_fc_psr_min', 0.763)
claim('nb04 app:sr:supp', 'gp_fc_psr_max',      'nb04', 'gp_fc_psr_max', 0.825)
claim('nb04 app:sr:supp', 'gp_star_psr_min',    'nb04', 'gp_star_psr_min', 0.733)
claim('nb04 app:sr:supp', 'gp_star_psr_max',    'nb04', 'gp_star_psr_max', 0.809)
claim('nb04 app:sr:supp', 'hi_fc_psr_min',      'nb04', 'hi_fc_psr_min',  0.841)
claim('nb04 app:sr:supp', 'hi_fc_psr_max',      'nb04', 'hi_fc_psr_max',  0.883)
claim('nb04 app:sr:supp', 'hi_star_psr_min',    'nb04', 'hi_star_psr_min', 0.806)
claim('nb04 app:sr:supp', 'hi_star_psr_max',    'nb04', 'hi_star_psr_max', 0.847)

claim('nb04 app:sr:supp', 'slope_acc_r_gpqa_fc',   'nb04', 'slope_acc_r_gpqa_fc',   0.02, tol=0.005)
claim('nb04 app:sr:supp', 'slope_acc_r_gpqa_star',  'nb04', 'slope_acc_r_gpqa_star',  0.04, tol=0.005)
claim('nb04 app:sr:supp', 'slope_acc_r_hb_fc',     'nb04', 'slope_acc_r_hb_fc',    -0.21, tol=0.005)
claim('nb04 app:sr:supp', 'slope_acc_r_hb_star',   'nb04', 'slope_acc_r_hb_star',  -0.31, tol=0.005)

claim('nb04 tab:sr:subgroups', 'sg_no_ceiling_n',      'nb04', 'sg_no_ceiling_n',      115137, tol=0)
claim('nb04 tab:sr:subgroups', 'sg_no_ceiling_slope',  'nb04', 'sg_no_ceiling_slope',   0.404)
claim('nb04 tab:sr:subgroups', 'sg_no_ceiling_psr',    'nb04', 'sg_no_ceiling_psr',     0.798)
claim('nb04 tab:sr:subgroups', 'sg_ceiling_hit_n',     'nb04', 'sg_ceiling_hit_n',      18559, tol=0)
claim('nb04 tab:sr:subgroups', 'sg_ceiling_hit_slope', 'nb04', 'sg_ceiling_hit_slope',  0.564)
claim('nb04 tab:sr:subgroups', 'sg_ceiling_hit_psr',   'nb04', 'sg_ceiling_hit_psr',    0.910)
claim('nb04 tab:sr:subgroups', 'sg_non_terminal_n',    'nb04', 'sg_non_terminal_n',     14341, tol=0)
claim('nb04 tab:sr:subgroups', 'sg_non_terminal_slope','nb04', 'sg_non_terminal_slope', 0.262)
claim('nb04 tab:sr:subgroups', 'sg_non_terminal_psr',  'nb04', 'sg_non_terminal_psr',   0.682)
claim('nb04 tab:sr:subgroups', 'sg_terminal_n',        'nb04', 'sg_terminal_n',        119355, tol=0)
claim('nb04 tab:sr:subgroups', 'sg_terminal_slope',    'nb04', 'sg_terminal_slope',     0.445)
claim('nb04 tab:sr:subgroups', 'sg_terminal_psr',      'nb04', 'sg_terminal_psr',       0.829)
claim('nb04 tab:sr:subgroups', 'sg_init_wrong_n',      'nb04', 'sg_init_wrong_n',      116400, tol=0)
claim('nb04 tab:sr:subgroups', 'sg_init_wrong_slope',  'nb04', 'sg_init_wrong_slope',   0.410)
claim('nb04 tab:sr:subgroups', 'sg_init_wrong_psr',    'nb04', 'sg_init_wrong_psr',     0.809)
claim('nb04 tab:sr:subgroups', 'sg_init_correct_n',    'nb04', 'sg_init_correct_n',     17296, tol=0)
claim('nb04 tab:sr:subgroups', 'sg_init_correct_slope','nb04', 'sg_init_correct_slope', 0.535)
claim('nb04 tab:sr:subgroups', 'sg_init_correct_psr',  'nb04', 'sg_init_correct_psr',   0.840)
claim('nb04 tab:sr:subgroups', 'sg_flip_n',            'nb04', 'sg_flip_n',            113660, tol=0)
claim('nb04 tab:sr:subgroups', 'sg_flip_slope',        'nb04', 'sg_flip_slope',         0.380)
claim('nb04 tab:sr:subgroups', 'sg_flip_psr',          'nb04', 'sg_flip_psr',           0.796)
claim('nb04 tab:sr:subgroups', 'sg_never_flip_n',      'nb04', 'sg_never_flip_n',       20036, tol=0)
claim('nb04 tab:sr:subgroups', 'sg_never_flip_slope',  'nb04', 'sg_never_flip_slope',   0.685)
claim('nb04 tab:sr:subgroups', 'sg_never_flip_psr',    'nb04', 'sg_never_flip_psr',     0.908)
claim('nb04 tab:sr:subgroups', 'sg_dist_40000_n',      'nb04', 'sg_dist_40000_n',       21435, tol=0)
claim('nb04 tab:sr:subgroups', 'sg_dist_40000_slope',  'nb04', 'sg_dist_40000_slope',   0.663)
claim('nb04 tab:sr:subgroups', 'sg_dist_40000_psr',    'nb04', 'sg_dist_40000_psr',     0.897)
claim('nb04 tab:sr:subgroups', 'sg_dist_31000_n',      'nb04', 'sg_dist_31000_n',       50286, tol=0)
claim('nb04 tab:sr:subgroups', 'sg_dist_31000_slope',  'nb04', 'sg_dist_31000_slope',   0.412)
claim('nb04 tab:sr:subgroups', 'sg_dist_31000_psr',    'nb04', 'sg_dist_31000_psr',     0.824)
claim('nb04 tab:sr:subgroups', 'sg_dist_22000_n',      'nb04', 'sg_dist_22000_n',       26629, tol=0)
claim('nb04 tab:sr:subgroups', 'sg_dist_22000_slope',  'nb04', 'sg_dist_22000_slope',   0.370)
claim('nb04 tab:sr:subgroups', 'sg_dist_22000_psr',    'nb04', 'sg_dist_22000_psr',     0.791)
claim('nb04 tab:sr:subgroups', 'sg_dist_21100_n',      'nb04', 'sg_dist_21100_n',       34035, tol=0)
claim('nb04 tab:sr:subgroups', 'sg_dist_21100_slope',  'nb04', 'sg_dist_21100_slope',   0.345)
claim('nb04 tab:sr:subgroups', 'sg_dist_21100_psr',    'nb04', 'sg_dist_21100_psr',     0.767)
claim('nb04 tab:sr:subgroups', 'sg_dist_11110_n',      'nb04', 'sg_dist_11110_n',        1311, tol=0)
claim('nb04 tab:sr:subgroups', 'sg_dist_11110_slope',  'nb04', 'sg_dist_11110_slope',   0.308)
claim('nb04 tab:sr:subgroups', 'sg_dist_11110_psr',    'nb04', 'sg_dist_11110_psr',     0.656)


# ─── nb05: Bistability ────────────────────────────────────────────────────────

exact('nb05 sec:results:multistability', 'n_multistable',   'nb05', 'n_multistable',   249)
exact('nb05 sec:results:multistability', 'n_stochastic',    'nb05', 'n_stochastic',    209)
exact('nb05 sec:results:multistability', 'n_monostable',    'nb05', 'n_monostable',    142)
claim('nb05 sec:results:multistability', 'frac_eff_gt15',   'nb05', 'frac_eff_gt15',   0.76, tol=0.005)

exact('nb05 app:multistab', 'gpqa_n_monostable',   'nb05', 'gpqa_n_monostable',   86)
exact('nb05 app:multistab', 'gpqa_n_multistable',  'nb05', 'gpqa_n_multistable', 151)
exact('nb05 app:multistab', 'gpqa_n_stochastic',   'nb05', 'gpqa_n_stochastic',   63)
exact('nb05 app:multistab', 'hb_n_monostable',     'nb05', 'hb_n_monostable',     56)
exact('nb05 app:multistab', 'hb_n_multistable',    'nb05', 'hb_n_multistable',    98)
exact('nb05 app:multistab', 'hb_n_stochastic',     'nb05', 'hb_n_stochastic',    146)

claim('nb05 app:multistab', 'gpqa_neff_mean',   'nb05', 'gpqa_neff_mean',   1.97, tol=0.005)
claim('nb05 app:multistab', 'gpqa_v_mean',      'nb05', 'gpqa_v_mean',      0.45, tol=0.005)
claim('nb05 app:multistab', 'hb_neff_mean',     'nb05', 'hb_neff_mean',     2.06, tol=0.005)
claim('nb05 app:multistab', 'hb_v_mean',        'nb05', 'hb_v_mean',        0.35, tol=0.005)
claim('nb05 app:multistab', 'cfg_neff_min',     'nb05', 'cfg_neff_min',     1.93, tol=0.005)
claim('nb05 app:multistab', 'cfg_neff_max',     'nb05', 'cfg_neff_max',     2.12, tol=0.005)
claim('nb05 app:multistab', 'cfg_v_min',        'nb05', 'cfg_v_min',        0.33, tol=0.005)
claim('nb05 app:multistab', 'cfg_v_max',        'nb05', 'cfg_v_max',        0.50, tol=0.005)

claim('nb05 sec:results:multistability', 'neff_acc_rho_gpqa_task', 'nb05', 'neff_acc_rho_gpqa_task', -0.16, tol=0.005)
claim('nb05 sec:results:multistability', 'neff_acc_rho_hb_task',   'nb05', 'neff_acc_rho_hb_task',   +0.44, tol=0.005)
claim('nb05 sec:results:multistability', 'neff_sr_rho_gpqa_task',  'nb05', 'neff_sr_rho_gpqa_task',  -0.74, tol=0.005)
claim('nb05 sec:results:multistability', 'neff_sr_rho_hb_task',    'nb05', 'neff_sr_rho_hb_task',    -0.44, tol=0.005)


# ─── nb06: Dominance ─────────────────────────────────────────────────────────

exact('nb06 sec:results:dominance', 'n_tasks_gpqa_pos', 'nb06', 'n_tasks_gpqa_pos', 49)
exact('nb06 sec:results:dominance', 'n_tasks_hb_pos',   'nb06', 'n_tasks_hb_pos',   50)

claim('nb06 app:dom:supp', 'gpqa_fc_z_bar', 'nb06', 'gpqa_fc_z_bar',  0.53, tol=0.005)
claim('nb06 app:dom:supp', 'hb_fc_z_bar',   'nb06', 'hb_fc_z_bar',    0.73, tol=0.005)
claim('nb06 app:dom:supp', 'gpqa_fc_flagged_pct', 'nb06', 'gpqa_fc_flagged_pct', 7.7)
claim('nb06 app:dom:supp', 'hb_fc_flagged_pct',
      'nb06', 'hb_fc_flagged_pct', 11.2, tol=0.1)

claim('nb06 sec:results:dominance', 'drounds_rho_gpqa', 'nb06', 'drounds_rho_gpqa_median', 0.60, tol=0.005)
claim('nb06 sec:results:dominance', 'drounds_rho_hb',   'nb06', 'drounds_rho_hb_median',   0.66, tol=0.005)

claim('nb06 app:dom:supp', 'gp_fc_w1_D',   'nb06', 'gp_fc_w1_D',   0.154)
claim('nb06 app:dom:supp', 'gp_fc_w2_D',   'nb06', 'gp_fc_w2_D',   0.154)
claim('nb06 app:dom:supp', 'gp_fc_w5_D',   'nb06', 'gp_fc_w5_D',   0.155)
claim('nb06 app:dom:supp', 'gp_star_w1_D', 'nb06', 'gp_star_w1_D', 0.601)
claim('nb06 app:dom:supp', 'gp_star_w2_D', 'nb06', 'gp_star_w2_D', 0.601)
claim('nb06 app:dom:supp', 'gp_star_w5_D', 'nb06', 'gp_star_w5_D', 0.610)
claim('nb06 app:dom:supp', 'hi_fc_w1_D',   'nb06', 'hi_fc_w1_D',   0.158)
claim('nb06 app:dom:supp', 'hi_fc_w2_D',   'nb06', 'hi_fc_w2_D',   0.178)
claim('nb06 app:dom:supp', 'hi_fc_w5_D',   'nb06', 'hi_fc_w5_D',   0.179)
claim('nb06 app:dom:supp', 'hi_star_w1_D', 'nb06', 'hi_star_w1_D', 0.540)
claim('nb06 app:dom:supp', 'hi_star_w2_D', 'nb06', 'hi_star_w2_D', 0.574)
claim('nb06 app:dom:supp', 'hi_star_w5_D', 'nb06', 'hi_star_w5_D', 0.578)


# ─── nb07: Limit Cycles ───────────────────────────────────────────────────────

exact('nb07 sec:results:lc', 'n_agent_total',      'nb07', 'n_agent_total',    120000)
exact('nb07 sec:results:lc', 'n_agent_fp',         'nb07', 'n_agent_fp',       119355)
exact('nb07 sec:results:lc', 'n_sys_total',        'nb07', 'n_sys_total',       30000)
exact('nb07 sec:results:lc', 'n_sys_fp',           'nb07', 'n_sys_fp',          29609)
exact('nb07 sec:results:lc', 'n_agent_candidates', 'nb07', 'n_agent_candidates',  633)
exact('nb07 sec:results:lc', 'n_sys_candidates',   'nb07', 'n_sys_candidates',    388)
exact('nb07 sec:results:lc', 'n_agent_too_short',  'nb07', 'n_agent_too_short',    12)
exact('nb07 sec:results:lc', 'n_sys_too_short',    'nb07', 'n_sys_too_short',       3)
exact('nb07 sec:results:lc', 'n_agent_flagged',    'nb07', 'n_agent_flagged',      47)
exact('nb07 sec:results:lc', 'n_sys_flagged',      'nb07', 'n_sys_flagged',        32)
exact('nb07 sec:results:lc', 'n_genuine_p2_agent', 'nb07', 'n_genuine_p2_agent',   33)
exact('nb07 sec:results:lc', 'n_agent_flag_star',  'nb07', 'n_agent_flag_star',    45)
exact('nb07 sec:results:lc', 'n_agent_flag_hb',    'nb07', 'n_agent_flag_hb',      34)
exact('nb07 sec:results:lc', 'n_sys_flag_star',    'nb07', 'n_sys_flag_star',       26)

claim('nb07 sec:results:lc', 'pct_agent_fp',          'nb07', 'pct_agent_fp',         99.5)
claim('nb07 sec:results:lc', 'pct_sys_fp',            'nb07', 'pct_sys_fp',            98.7)
claim('nb07 sec:results:lc', 'pct_genuine_p2_agent%', 'nb07', 'pct_genuine_p2_agent', 70.0)

claim('nb07 app:lc:supp', 'p_lc_cond_agent',     'nb07', 'p_lc_cond_agent',    0.074)
claim('nb07 app:lc:supp', 'p_lc_cond_sys',       'nb07', 'p_lc_cond_sys',      0.083)
claim('nb07 app:lc:supp', 'fp_floor_agent_pct',  'nb07', 'fp_floor_agent_pct', 0.92)
claim('nb07 app:lc:supp', 'fp_floor_sys_pct',    'nb07', 'fp_floor_sys_pct',   0.71)


# ─── nb08: High-Repetition ────────────────────────────────────────────────────

claim('nb08 tab:highrep:init_votes', 'q84_acc',           'nb08', 'tasks.q84.acc',           0.269)
exact('nb08 tab:highrep:init_votes', 'q84_n_absent',       'nb08', 'tasks.q84.n_absent',       257)
claim('nb08 tab:highrep:init_votes', 'q84_acc_absent',    'nb08', 'tasks.q84.acc_absent',     0.004)
exact('nb08 tab:highrep:init_votes', 'q84_n_present',      'nb08', 'tasks.q84.n_present',      743)
claim('nb08 tab:highrep:init_votes', 'q84_acc_present',   'nb08', 'tasks.q84.acc_present',    0.361)
exact('nb08 tab:highrep:init_votes', 'q84_n_plurality',    'nb08', 'tasks.q84.n_plurality',    213)
claim('nb08 tab:highrep:init_votes', 'q84_acc_plurality', 'nb08', 'tasks.q84.acc_plurality',  0.620)
claim('nb08 tab:highrep:init_votes', 'q84_acc_present_nonplur', 'nb08', 'tasks.q84.acc_present_nonplur', 0.257)

claim('nb08 tab:highrep:init_votes', 'q125_acc',           'nb08', 'tasks.q125.acc',          0.273)
exact('nb08 tab:highrep:init_votes', 'q125_n_absent',       'nb08', 'tasks.q125.n_absent',      293)
claim('nb08 tab:highrep:init_votes', 'q125_acc_absent',    'nb08', 'tasks.q125.acc_absent',    0.031)
exact('nb08 tab:highrep:init_votes', 'q125_n_present',      'nb08', 'tasks.q125.n_present',     707)
claim('nb08 tab:highrep:init_votes', 'q125_acc_present',   'nb08', 'tasks.q125.acc_present',   0.373)
exact('nb08 tab:highrep:init_votes', 'q125_n_plurality',    'nb08', 'tasks.q125.n_plurality',   250)
claim('nb08 tab:highrep:init_votes', 'q125_acc_plurality', 'nb08', 'tasks.q125.acc_plurality', 0.552)

claim('nb08 tab:highrep:init_votes', 'q144_acc',           'nb08', 'tasks.q144.acc',          0.509)
exact('nb08 tab:highrep:init_votes', 'q144_n_absent',       'nb08', 'tasks.q144.n_absent',      144)
claim('nb08 tab:highrep:init_votes', 'q144_acc_absent',    'nb08', 'tasks.q144.acc_absent',    0.069)
exact('nb08 tab:highrep:init_votes', 'q144_n_present',      'nb08', 'tasks.q144.n_present',     856)
claim('nb08 tab:highrep:init_votes', 'q144_acc_present',   'nb08', 'tasks.q144.acc_present',   0.583)
exact('nb08 tab:highrep:init_votes', 'q144_n_plurality',    'nb08', 'tasks.q144.n_plurality',   403)
claim('nb08 tab:highrep:init_votes', 'q144_acc_plurality', 'nb08', 'tasks.q144.acc_plurality', 0.717)


# ─── nb09: Seeded Causal ─────────────────────────────────────────────────────

claim('nb09 sec:results:seeded', 'seeded_k0_acc',              'nb09', 'seeded_k0_acc',              0.018)
claim('nb09 sec:results:seeded', 'seeded_k1_acc',              'nb09', 'seeded_k1_acc',              0.262)
claim('nb09 sec:results:seeded', 'seeded_k2_acc',              'nb09', 'seeded_k2_acc',              0.637)
claim('nb09 sec:results:seeded', 'seeded_k3_acc',              'nb09', 'seeded_k3_acc',              0.863)
exact('nb09 sec:results:seeded', 'seeded_k0_to_k1_n_tasks',    'nb09', 'seeded_k0_to_k1_increase_n_tasks', 50)
exact('nb09 sec:results:seeded', 'seeded_k1_to_k2_n_tasks',    'nb09', 'seeded_k1_to_k2_increase_n_tasks', 50)
exact('nb09 sec:results:seeded', 'seeded_k2_to_k3_n_tasks',    'nb09', 'seeded_k2_to_k3_increase_n_tasks', 47)
exact('nb09 sec:results:seeded', 'seeded_k0_denovo_tasks',     'nb09', 'seeded_k0_denovo_tasks',          21)
claim('nb09 sec:results:seeded', 'mean_first_round_k1_winners','nb09', 'mean_first_round_k1_winners',    1.3, tol=0.05)
claim('nb09 sec:results:seeded', 'seeded_k2_plurality_baseline','nb09', 'seeded_k2_plurality_baseline',  0.38, tol=0.005)
claim('nb09 sec:results:seeded', 'da_k2_base_acc',             'nb09', 'da_k2_base_acc',                0.637)
claim('nb09 sec:results:seeded', 'da_acc',                     'nb09', 'da_acc',                        0.100)
exact('nb09 sec:results:seeded', 'da_drop_n_tasks',            'nb09', 'da_drop_n_tasks',                  50)


# ─── nb12: Effective Topology ─────────────────────────────────────────────────

claim('nb12 tab:efftopo:types', 'gpqa_flat_pct',       'nb12', 'gpqa_flat_pct',         5.4)
claim('nb12 tab:efftopo:types', 'gpqa_flat_acc',       'nb12', 'gpqa_flat_acc',         0.394)
claim('nb12 tab:efftopo:types', 'gpqa_hub_pct',        'nb12', 'gpqa_hub_pct',         11.0)
claim('nb12 tab:efftopo:types', 'gpqa_hub_acc',        'nb12', 'gpqa_hub_acc',          0.393)
claim('nb12 tab:efftopo:types', 'gpqa_two-hub_pct',    'nb12', 'gpqa_two-hub_pct',     58.7)
claim('nb12 tab:efftopo:types', 'gpqa_two-hub_acc',    'nb12', 'gpqa_two-hub_acc',      0.418)
claim('nb12 tab:efftopo:types', 'gpqa_star_pct',       'nb12', 'gpqa_star_pct',         2.3)
claim('nb12 tab:efftopo:types', 'gpqa_star_acc',       'nb12', 'gpqa_star_acc',         0.485)
claim('nb12 tab:efftopo:types', 'gpqa_degenerate_pct', 'nb12', 'gpqa_degenerate_pct',  22.6)
claim('nb12 tab:efftopo:types', 'gpqa_degenerate_acc', 'nb12', 'gpqa_degenerate_acc',  0.450)

claim('nb12 tab:efftopo:types', 'hb_flat_pct',         'nb12', 'hiddenbench_flat_pct',       6.8)
claim('nb12 tab:efftopo:types', 'hb_flat_acc',         'nb12', 'hiddenbench_flat_acc',       0.242)
claim('nb12 tab:efftopo:types', 'hb_hub_pct',          'nb12', 'hiddenbench_hub_pct',       18.0)
claim('nb12 tab:efftopo:types', 'hb_hub_acc',          'nb12', 'hiddenbench_hub_acc',        0.377)
claim('nb12 tab:efftopo:types', 'hb_two-hub_pct',      'nb12', 'hiddenbench_two-hub_pct',   59.2)
claim('nb12 tab:efftopo:types', 'hb_two-hub_acc',      'nb12', 'hiddenbench_two-hub_acc',    0.247)
claim('nb12 tab:efftopo:types', 'hb_star_pct',         'nb12', 'hiddenbench_star_pct',        5.4)
claim('nb12 tab:efftopo:types', 'hb_star_acc',         'nb12', 'hiddenbench_star_acc',        0.562)
claim('nb12 tab:efftopo:types', 'hb_degenerate_pct',   'nb12', 'hiddenbench_degenerate_pct', 10.6)
claim('nb12 tab:efftopo:types', 'hb_degenerate_acc',   'nb12', 'hiddenbench_degenerate_acc', 0.029)

claim('nb12 sec:results:eff_topology', 'gpqa_overall_fc_acc', 'nb12', 'gpqa_overall_fc_acc', 0.423)
claim('nb12 sec:results:eff_topology', 'hb_overall_fc_acc',   'nb12', 'hb_overall_fc_acc',   0.264)

claim('nb12 sec:results:eff_topology', 'gpqa_star_hub_correct_pct', 'nb12', 'gpqa_star_hub_correct_rate_cond', 0.545, tol=0.005)
claim('nb12 sec:results:eff_topology', 'hb_star_hub_correct_pct',   'nb12', 'hiddenbench_star_hub_correct_rate_cond', 0.855, tol=0.005)
claim('nb12 sec:results:eff_topology', 'gpqa_hub_correct_to_acc',   'nb12', 'gpqa_hub_correct_to_acc',  0.861, tol=0.005)
claim('nb12 sec:results:eff_topology', 'gpqa_hub_wrong_to_acc',     'nb12', 'gpqa_hub_wrong_to_acc',    0.156, tol=0.005)
claim('nb12 sec:results:eff_topology', 'hb_hub_correct_to_acc',     'nb12', 'hiddenbench_hub_correct_to_acc', 0.924, tol=0.005)
claim('nb12 sec:results:eff_topology', 'hb_hub_wrong_to_acc',       'nb12', 'hiddenbench_hub_wrong_to_acc',   0.107, tol=0.005)


# ─── runner ───────────────────────────────────────────────────────────────────

def run():
    data = {
        'nb03': _load('03_system_performance'),
        'nb04': _load('04_self_reinforcement'),
        'nb05': _load('05_bistability'),
        'nb06': _load('06_dominance'),
        'nb07': _load('07_limit_cycles'),
        'nb08': _load('08_high_repetition'),
        'nb09': _load('09_seeded_causal'),
        'nb12': _load('12_effective_topology'),
    }

    ok = fail = warn = 0
    current_section = None

    for section, label, nb, path, expected, tol, note in CLAIMS:
        if section != current_section:
            print(f'\n=== {section} ===')
            current_section = section

        d = data.get(nb)
        val = _get(d, path)

        if val is None:
            status = 'MISSING'
            warn += 1
        elif tol == 0:
            match = val == expected
            if match:
                status = 'OK'
                ok += 1
            elif note:
                status = 'WARN'
                warn += 1
            else:
                status = 'FAIL'
                fail += 1
        else:
            match = abs(float(val) - float(expected)) <= tol
            if match:
                status = 'OK'
                ok += 1
            elif note:
                status = 'WARN'
                warn += 1
            else:
                status = 'FAIL'
                fail += 1

        suffix = f'  [{note}]' if (note and status != 'OK') else ''
        if status == 'OK':
            print(f'  OK    {label:<45}  computed={val}  expected={expected}')
        else:
            print(f'  {status:<6}{label:<45}  computed={val}  expected={expected}{suffix}')

    print(f'\n{"="*70}')
    print(f'TOTAL: {ok} OK  |  {fail} FAIL  |  {warn} WARN')
    if fail > 0:
        print('RESULT: FAILED')
    elif warn > 0:
        print('RESULT: PASSED WITH WARNINGS')
    else:
        print('RESULT: PASSED')
    return fail == 0


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
