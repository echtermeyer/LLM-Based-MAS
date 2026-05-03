"""
Multi-Agent System Run Dashboard
Usage:
    streamlit run dashboard.py -- results/mas_eval/<file>.json
    streamlit run dashboard.py   # sidebar file picker
"""

import json
import math
import pathlib
import sys
import time

import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be first st.* call
# ---------------------------------------------------------------------------

st.set_page_config(
    layout="wide",
    page_title="MAS Run Dashboard",
    page_icon=":material/hub:",
)

RESULTS_DIR = pathlib.Path(__file__).parent / "results" / "mas_eval"

OPTION_COLORS = {
    "A": "#1f77b4",   # blue
    "B": "#ff7f0e",   # orange
    "C": "#2ca02c",   # green
    "D": "#9467bd",   # purple  (not red — red is reserved for correct/wrong borders)
}
OPTIONS = ["A", "B", "C", "D"]

COLOR_CORRECT = "#00cc44"
COLOR_WRONG = "#cc2200"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_run(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# File selection + run metadata + playback controls (sidebar)
# ---------------------------------------------------------------------------

_cli_arg = sys.argv[1] if len(sys.argv) > 1 else None

json_files = sorted(RESULTS_DIR.glob("*.json"))
file_names = [f.name for f in json_files]

with st.sidebar:
    st.header("Run selection")
    if not file_names:
        st.error(f"No JSON files found in {RESULTS_DIR}")
        st.stop()
    default_idx = len(file_names) - 1
    if _cli_arg:
        cli_name = pathlib.Path(_cli_arg).name
        if cli_name in file_names:
            default_idx = file_names.index(cli_name)
    selected_name = st.selectbox("Select run file", file_names, index=default_idx)
    file_path = str(RESULTS_DIR / selected_name)

data = load_run(file_path)

# ---------------------------------------------------------------------------
# Derived statics (computed once per file, reused across slider ticks)
# ---------------------------------------------------------------------------

N = data["N"]
T = data["T"]
R = T + 1
GT = data["ground_truth"]

# votes[r][i] = vote of agent i at round r
votes: dict[int, dict[int, str]] = {
    entry["round"]: {e["id"]: e["vote"] for e in entry["phase_b"]}
    for entry in data["trajectory"]
}

# Fraction of agents holding each option at each round
fracs: dict[str, list[float]] = {
    opt: [
        sum(1 for i in range(N) if votes[r].get(i) == opt) / N
        for r in range(R)
    ]
    for opt in OPTIONS
}

scalars = data["scalars"]
per_round_sc = scalars["per_round"]
per_agent_sc = scalars["per_agent_per_round"]
per_pair_sc = scalars["per_pair_per_round"]

entropy_vals: list[float] = [max(0.0, v) for v in per_round_sc["entropy"]]
mean_correct_vals: list[float] = per_round_sc["mean_correct"]
mean_diss_vals: list = per_round_sc.get("mean_diss", [None] * R)

diss: list[list] = per_agent_sc["diss"]       # diss[i][r]
sim_pub: list[list[list]] = per_pair_sc["sim_pub"]  # sim_pub[i][j][r]
toward_sc: list[list[list]] = per_pair_sc["toward"]  # toward[i][j][r]

# Global sim_pub range for fixed heatmap color scale
_all_sim = [
    sim_pub[i][j][r]
    for i in range(N)
    for j in range(N)
    for r in range(R)
    if i != j and sim_pub[i][j][r] is not None
]
SIM_PUB_VMIN = min(_all_sim) if _all_sim else 0.5
SIM_PUB_VMAX = max(_all_sim) if _all_sim else 1.0

# Global diss max for fixed bar chart y-axis
_all_diss = [
    diss[i][r]
    for i in range(N)
    for r in range(R)
    if diss[i][r] is not None
]
DISS_YMAX = max(_all_diss) * 1.15 if _all_diss else 1.0

# Global toward max for fixed directed-edge width scale
_all_toward_pos = [
    toward_sc[i][j][r]
    for i in range(N)
    for j in range(N)
    for r in range(R)
    if i != j and toward_sc[i][j][r] is not None and toward_sc[i][j][r] > 0
]
TOWARD_VMAX = max(_all_toward_pos) if _all_toward_pos else 0.01

# px → data-unit conversion: figure height 380px, margins top=50 bottom=60 → 270px plot area,
# y-range = 3.2 units; scaleanchor makes x match y scale.
PX_TO_DATA = 3.2 / 270  # ≈ 0.01185 data units / px

NODE_SIZE_BASE = 32   # px — node with zero received influence
NODE_SIZE_MAX  = 64   # px — node with max received influence

# Global max of total positive influence received (across all agents & rounds)
_all_received = [
    sum(
        toward_sc[i][j][r]
        for i in range(N)
        if i != j and toward_sc[i][j][r] is not None and toward_sc[i][j][r] > 0
    )
    for j in range(N)
    for r in range(R)
]
_RECEIVED_MAX = max(_all_received) if any(v > 0 for v in _all_received) else 0.01

# Node layout: circular, top-anchored
node_x = [math.cos(2 * math.pi * i / N - math.pi / 2) for i in range(N)]
node_y = [math.sin(2 * math.pi * i / N - math.pi / 2) for i in range(N)]

# Model display name
raw_model = data["agents"][0]["model"]
model_name = raw_model.split("--")[-1] if "--" in raw_model else raw_model

topology = data["topology"]

# ---------------------------------------------------------------------------
# Sidebar: run metadata + playback settings
# ---------------------------------------------------------------------------

with st.sidebar:
    with st.container(border=True):
        st.markdown(
            f"**Run:** `{data['run_id'][:8]}…`  \n"
            f"**Question:** {data['question_id']}  \n"
            f"**Ground truth:** `{GT}`  \n"
            f"**Agents (N):** {N}  \n"
            f"**Rounds (T):** {T}  \n"
            f"**Model:** {model_name}"
        )

    st.header("Playback")
    speed = st.slider("Speed (s/round)", 0.2, 2.0, 0.8, 0.1)
    loop = st.toggle("Loop", value=True)


# ---------------------------------------------------------------------------
# Edge width helper
# ---------------------------------------------------------------------------

def _scale_width(v, vmin, vmax, lo=1.0, hi=5.0):
    if vmax == vmin:
        return (lo + hi) / 2
    return lo + (hi - lo) * (v - vmin) / (vmax - vmin)


def _node_sizes(r: int) -> list[float]:
    """Node size = total positive influence received at round r."""
    received = [
        sum(
            toward_sc[i][j][r]
            for i in range(N)
            if i != j and toward_sc[i][j][r] is not None and toward_sc[i][j][r] > 0
        )
        for j in range(N)
    ]
    if _RECEIVED_MAX == 0:
        return [40.0] * N
    return [
        NODE_SIZE_BASE + (NODE_SIZE_MAX - NODE_SIZE_BASE) * (v / _RECEIVED_MAX)
        for v in received
    ]


# ---------------------------------------------------------------------------
# Network graph builder
# ---------------------------------------------------------------------------

def build_network_fig(r: int) -> go.Figure:
    fig = go.Figure()
    node_sizes = _node_sizes(r)

    # Draw directed edges: i→j arrow width = positive toward[i][j][r] only
    for i in range(N):
        for j in range(N):
            if i == j or not topology[i][j]:
                continue
            t = toward_sc[i][j][r]
            if t is None or t <= 0:
                continue
            width = _scale_width(t, 0.0, TOWARD_VMAX, lo=0.8, hi=5.0)
            alpha = 0.3 + 0.6 * (t / TOWARD_VMAX)
            dx = node_x[j] - node_x[i]
            dy = node_y[j] - node_y[i]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist == 0:
                continue
            ux, uy = dx / dist, dy / dist
            # offset = marker radius + half border width, converted to data coords
            off_j = (node_sizes[j] / 2 + 2) * PX_TO_DATA
            off_i = (node_sizes[i] / 2 + 2) * PX_TO_DATA
            fig.add_annotation(
                x=node_x[j] - ux * off_j,
                y=node_y[j] - uy * off_j,
                ax=node_x[i] + ux * off_i,
                ay=node_y[i] + uy * off_i,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=2, arrowsize=1,
                arrowwidth=width,
                arrowcolor=f"rgba(180,180,180,{alpha:.2f})",
                text="",
            )

    # Draw nodes
    node_colors = [OPTION_COLORS[votes[r][i]] for i in range(N)]
    border_colors = [COLOR_CORRECT if votes[r][i] == GT else COLOR_WRONG for i in range(N)]
    received_sums = [
        sum(
            toward_sc[i][j][r]
            for i in range(N)
            if i != j and toward_sc[i][j][r] is not None and toward_sc[i][j][r] > 0
        )
        for j in range(N)
    ]
    hover_texts = [
        f"Agent {j}<br>Vote: {votes[r][j]}<br>{'CORRECT' if votes[r][j] == GT else 'Wrong'}"
        f"<br>∑toward→j: {received_sums[j]:.3f}"
        for j in range(N)
    ]
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(color=border_colors, width=4),
        ),
        text=[f"A{i}" for i in range(N)],
        textposition="middle center",
        textfont=dict(color="white", size=12, family="monospace"),
        hovertext=hover_texts,
        hoverinfo="text",
        showlegend=False,
    ))

    # Legend entries for correctness borders
    for label, color in [("Correct", COLOR_CORRECT), ("Wrong", COLOR_WRONG)]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=12, color="rgba(0,0,0,0)", line=dict(color=color, width=3)),
            name=label,
            showlegend=True,
        ))

    # Legend entries for option colors
    for opt in OPTIONS:
        suffix = " (GT)" if opt == GT else ""
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=10, color=OPTION_COLORS[opt]),
            name=f"Belief {opt}{suffix}",
            showlegend=True,
        ))

    padding = 1.6
    fig.update_layout(
        title=dict(text=f"Agent Network — Round {r}" + (" (no edges: toward undefined at r=0)" if r == 0 else ""), x=0.5, xanchor="center", xref="container"),
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center", font=dict(size=11)),
        xaxis=dict(range=[-padding, padding], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-padding, padding], showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x"),
        margin=dict(l=20, r=20, t=50, b=60),
        height=380,
    )
    return fig


# ---------------------------------------------------------------------------
# 2×2 subplot grid builder
# ---------------------------------------------------------------------------

def _base_layout(title: str, xlab: str, ylab: str, height: int = 320) -> dict:
    rounds_axis = list(range(R))
    return dict(
        title=dict(text=title, x=0.5, xanchor="center", xref="container", font=dict(size=14)),
        xaxis=dict(title=xlab, tickvals=rounds_axis, tickmode="array",
                   range=[-0.5, R - 0.5]),
        yaxis_title=ylab,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="center", x=0.5,
                    font=dict(size=11)),
        margin=dict(l=60, r=20, t=55, b=100),
        height=height,
    )


def build_vote_fig(r: int) -> go.Figure:
    rounds_axis = list(range(R))
    fig = go.Figure()
    for opt in OPTIONS:
        label = f"{opt}  ✓ (ground truth)" if opt == GT else opt
        fig.add_trace(go.Scatter(
            x=rounds_axis, y=fracs[opt],
            name=label,
            mode="lines",
            line=dict(width=0.5, color=OPTION_COLORS[opt]),
            fillcolor=OPTION_COLORS[opt],
            stackgroup="one",
        ))
    vline_pos = "top right" if r == 0 else "top left"
    fig.add_vline(x=r, line_dash="dash", line_color="white", opacity=0.7,
                  annotation_text=f"r={r}", annotation_position=vline_pos,
                  annotation_font_size=11)
    fig.update_layout(**_base_layout("Belief distribution", "Round", "Fraction of agents"))
    fig.update_yaxes(range=[0, 1])
    return fig


def build_convergence_fig(r: int) -> go.Figure:
    rounds_axis = list(range(R))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rounds_axis, y=entropy_vals,
        name="Entropy H(r)  —  0 = full consensus, log4 ≈ 1.39 = uniform",
        mode="lines+markers",
        line=dict(color="#9467bd", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=rounds_axis, y=mean_correct_vals,
        name="Fraction of agents holding the correct answer",
        mode="lines+markers",
        line=dict(color="#17becf", width=2, dash="dot"),
    ))
    # Reference lines as shapes + annotations to avoid clipping
    fig.add_shape(type="line", x0=0, x1=1, xref="paper",
                  y0=math.log(4), y1=math.log(4), yref="y",
                  line=dict(dash="dot", color="gray", width=1), opacity=0.5)
    fig.add_annotation(x=1, xref="paper", y=math.log(4), yref="y",
                       text="max entropy (log 4 ≈ 1.39)", showarrow=False,
                       xanchor="right", yanchor="bottom", font=dict(size=10, color="gray"))
    fig.add_shape(type="line", x0=0, x1=1, xref="paper",
                  y0=1.0, y1=1.0, yref="y",
                  line=dict(dash="dot", color="lightgray", width=1), opacity=0.4)
    fig.add_annotation(x=1, xref="paper", y=1.0, yref="y",
                       text="all correct (1.0)", showarrow=False,
                       xanchor="right", yanchor="bottom", font=dict(size=10, color="lightgray"))
    vline_pos = "top right" if r == 0 else "top left"
    fig.add_vline(x=r, line_dash="dash", line_color="white", opacity=0.7,
                  annotation_text=f"r={r}", annotation_position=vline_pos,
                  annotation_font_size=11)
    fig.update_layout(**_base_layout("Convergence metrics", "Round", "Value"))
    fig.update_yaxes(range=[0, 1.5])
    return fig


def build_heatmap_fig(r: int) -> go.Figure:
    z_mat = [
        [sim_pub[i][j][r] if i != j else None for j in range(N)]
        for i in range(N)
    ]
    agent_labels = [f"Agent {i}" for i in range(N)]
    fig = go.Figure(go.Heatmap(
        z=z_mat,
        zmin=SIM_PUB_VMIN, zmax=SIM_PUB_VMAX,
        colorscale="Blues",
        showscale=True,
        xgap=2, ygap=2,
        x=agent_labels, y=agent_labels,
        hovertemplate="Agent %{x} × Agent %{y}<br>sim_pub = %{z:.4f}<extra></extra>",
        colorbar=dict(thickness=14, title=dict(text="cos sim", side="right"),
                      tickfont=dict(size=10)),
    ))
    fig.update_layout(
        title=dict(text=f"Public message similarity — Round {r}", x=0.5, xanchor="center",
                   xref="container", font=dict(size=14)),
        xaxis=dict(title="Agent j", side="bottom"),
        yaxis_title="Agent i",
        template="plotly_dark",
        margin=dict(l=70, r=80, t=55, b=60),
        height=320,
    )
    return fig


def build_diss_fig(r: int) -> go.Figure:
    agent_labels = [f"Agent {i}" for i in range(N)]
    diss_vals_r = [diss[i][r] if diss[i][r] is not None else 0.0 for i in range(N)]
    bar_colors = [OPTION_COLORS[votes[r][i]] for i in range(N)]
    fig = go.Figure(go.Bar(
        x=agent_labels, y=diss_vals_r,
        marker_color=bar_colors,
        hovertemplate="Agent %{x}<br>diss = %{y:.4f}<extra></extra>",
    ))
    mean_diss_r = mean_diss_vals[r] if r < len(mean_diss_vals) else None
    if mean_diss_r is not None:
        fig.add_shape(type="line", x0=0, x1=1, xref="paper",
                      y0=mean_diss_r, y1=mean_diss_r, yref="y",
                      line=dict(dash="dash", color="white", width=1), opacity=0.5)
        fig.add_annotation(x=1, xref="paper", y=mean_diss_r, yref="y",
                           text=f"mean = {mean_diss_r:.3f}", showarrow=False,
                           xanchor="right", yanchor="bottom",
                           font=dict(size=10, color="white"))
    fig.update_layout(
        title=dict(text=f"Dissociation — Round {r}", x=0.5, xanchor="center", xref="container",
                   font=dict(size=14)),
        xaxis_title="Agent",
        yaxis=dict(title="diss(i, r)", range=[0, DISS_YMAX]),
        template="plotly_dark",
        margin=dict(l=55, r=20, t=55, b=60),
        height=320,
    )
    return fig


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

# Question + options (full-width expander above main columns)
with st.expander(":material/quiz: Question & options", expanded=True):
    st.markdown(data["question"])
    st.markdown("**Options:**")
    for k, v in data["options"].items():
        gt_marker = " ✓" if k == GT else ""
        st.markdown(f"- **{k}{gt_marker}** {v.strip()}")

graph_col, plots_col = st.columns([2, 3])

with graph_col:
    # --- Playback state ---
    if "playing" not in st.session_state:
        st.session_state.playing = False
    if "current_round" not in st.session_state:
        st.session_state.current_round = T

    # Reset when a new file is loaded
    if st.session_state.get("loaded_file") != file_path:
        st.session_state.current_round = T
        st.session_state.playing = False
        st.session_state.loaded_file = file_path

    # Button LEFT, slider RIGHT — tight layout, no spacer
    btn_col, slider_col = st.columns([1, 3])
    with btn_col:
        label = ":material/stop:" if st.session_state.playing else ":material/play_arrow:"
        help_text = "Stop" if st.session_state.playing else "Play through all rounds"
        if st.button(label, help=help_text):
            if st.session_state.playing:
                st.session_state.playing = False
            else:
                st.session_state.current_round = 0
                st.session_state.playing = True
            st.rerun()
    with slider_col:
        current_round = st.slider(
            "Round", min_value=0, max_value=T,
            value=st.session_state.current_round, step=1,
            label_visibility="collapsed",
        )
        st.session_state.current_round = current_round  # sync manual drags

    # --- Network graph ---
    st.plotly_chart(build_network_fig(current_round), use_container_width=True)

with plots_col:
    row1_left, row1_right = st.columns(2)
    row2_left, row2_right = st.columns(2)

    with row1_left:
        st.plotly_chart(build_vote_fig(current_round), use_container_width=True)
        st.caption("p_a(r) = (1/N) Σ 1[vote_i(r) = a] — fraction of agents holding each answer at each round. Stacked areas sum to 1.")

    with row1_right:
        st.plotly_chart(build_convergence_fig(current_round), use_container_width=True)
        st.caption("H(r) = −Σ p_a log p_a (Shannon entropy; 0 = full consensus, log 4 ≈ 1.39 = uniform). Mean correct = fraction of agents with the right answer.")

    with row2_left:
        st.plotly_chart(build_heatmap_fig(current_round), use_container_width=True)
        st.caption("sim_pub(i,j,r) = cos(msg_i(r), msg_j(r)) — cosine similarity of agents' messages. Diagonal undefined (self-similarity). Color range fixed across all rounds.")

    with row2_right:
        st.plotly_chart(build_diss_fig(current_round), use_container_width=True)
        st.caption("diss(i,r) = 1 − cos(msg_i(r), reasoning_i(r)) — gap between an agent's message and its private reasoning. Bar color = agent's current vote.")

# --- Auto-advance (must be after all columns are rendered) ---
if st.session_state.playing:
    time.sleep(speed)
    next_round = (current_round + 1) % (T + 1)
    if not loop and current_round == T:
        st.session_state.playing = False
    else:
        st.session_state.current_round = next_round
    st.rerun()
