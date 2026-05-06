import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from src.mas.agent import Agent, PeerRecord, PhaseAOutput, PhaseBOutput, _format_phase_a
from src.mas.logging import PhaseAEntry, PhaseBEntry, RoundEntry
from src.mas.topology import neighbors


def run_round(
    agents: List[Agent],
    question_prompts: List[str],
    adjacency: List[List[int]],
    round_index: int,
    trajectory: List[RoundEntry],
    w: Optional[int],
) -> RoundEntry:
    if round_index == 0:
        phase_b_outputs = _run_parallel(
            agents,
            lambda agent: (agent.id, agent.init_round(question_prompts[agent.id])),
        )
        _flush_verbose(agents)
        return RoundEntry(
            round=0,
            phase_a=None,
            phase_b=_to_entries(PhaseBEntry, agents, phase_b_outputs),
        )

    phase_a_outputs: Dict[int, PhaseAOutput] = _run_parallel(
        agents,
        lambda agent: (
            agent.id,
            agent.phase_a(
                question_prompts[agent.id],
                _build_peer_window(
                    neighbors(adjacency, agent.id), trajectory, round_index, w, agents
                ),
            ),
        ),
    )
    _flush_verbose(agents)

    phase_b_outputs: Dict[int, PhaseBOutput] = _run_parallel(
        agents,
        lambda agent: (
            agent.id,
            agent.phase_b(
                question_prompts[agent.id],
                _format_phase_a(phase_a_outputs[agent.id]),
                _build_peer_window(
                    neighbors(adjacency, agent.id), trajectory, round_index, w, agents
                ),
                _shuffled_peer_drafts(
                    neighbors(adjacency, agent.id), agents, phase_a_outputs
                ),
            ),
        ),
    )
    _flush_verbose(agents)

    return RoundEntry(
        round=round_index,
        phase_a=_to_entries(PhaseAEntry, agents, phase_a_outputs),
        phase_b=_to_entries(PhaseBEntry, agents, phase_b_outputs),
    )


def _build_peer_window(
    neighbor_ids: List[int],
    trajectory: List[RoundEntry],
    round_index: int,
    w: Optional[int],
    agents: List[Agent],
) -> List[PeerRecord]:
    limit = round_index if w is None else min(round_index, w)
    records = []
    for k in range(1, limit + 1):
        r = round_index - k
        rentry = trajectory[r]
        pb_by_id = {e.id: e for e in rentry.phase_b}
        pa_by_id = {e.id: e for e in rentry.phase_a} if rentry.phase_a else {}
        peers_this_round = []
        for j in neighbor_ids:
            pb = pb_by_id[j]
            peers_this_round.append(
                PeerRecord(
                    name=agents[j].name,
                    round=r,
                    vote=pb.vote,
                    message=pb.message,
                    draft=(
                        f"defense: {pa_by_id[j].defense}\n"
                        f"challenge: {pa_by_id[j].challenge}\n"
                        f"question: {pa_by_id[j].question}"
                    ) if j in pa_by_id else None,
                )
            )
        random.shuffle(peers_this_round)
        records.extend(peers_this_round)
    return records


def _shuffled_peer_drafts(
    neighbor_ids: List[int],
    agents: List[Agent],
    phase_a_outputs: Dict[int, PhaseAOutput],
) -> List[Tuple[str, str]]:
    ns = list(neighbor_ids)
    random.shuffle(ns)
    return [(agents[j].name, _format_phase_a(phase_a_outputs[j])) for j in ns]


def _run_parallel(agents: List[Agent], fn) -> Dict[int, object]:
    results: Dict[int, object] = {}
    with ThreadPoolExecutor(max_workers=len(agents)) as executor:
        futures = {executor.submit(fn, agent): agent for agent in agents}
        for future in as_completed(futures):
            agent_id, output = future.result()
            results[agent_id] = output
    return results


def _to_entries(cls, agents: List[Agent], outputs: Dict) -> list:
    return [cls(id=a.id, **outputs[a.id].model_dump()) for a in agents]


def _flush_verbose(agents: List[Agent]) -> None:
    for agent in sorted(agents, key=lambda a: a.id):
        agent.flush_verbose()
