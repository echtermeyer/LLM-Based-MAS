from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from src.mas.agent import Agent, PhaseAOutput, PhaseBOutput
from src.mas.logging import PhaseAEntry, PhaseBEntry, RoundEntry
from src.mas.topology import neighbors


def run_round(
    agents: List[Agent],
    question_prompt: str,
    adjacency: List[List[int]],
    round_index: int,
    prev_phase_b_public_messages: Optional[List[str]],
) -> RoundEntry:
    """
    Execute one full round.

    Round 0: Phase B only — agents see the question and emit their initial belief.
    Round r≥1: Phase A (parallel) then Phase B (parallel).
      - Phase A receives neighbors' Phase B public_messages from round r-1.
      - Phase B receives neighbors' Phase A draft_messages from round r.

    Parameters
    ----------
    prev_phase_b_public_messages: public_messages from the previous round's Phase B,
                                  indexed by agent id. None on round 0.
    """
    if round_index == 0:
        phase_b_outputs = _run_parallel(
            agents,
            lambda agent: (agent.id, agent.respond_phase_b(0, question_prompt, [])),
        )
        return RoundEntry(
            round=0,
            phase_a=None,
            phase_b=_to_phase_b_entries(agents, phase_b_outputs),
        )

    # --- Phase A ---
    def call_phase_a(agent: Agent) -> Tuple[int, PhaseAOutput]:
        neighbor_msgs = [
            (j, prev_phase_b_public_messages[j])
            for j in neighbors(adjacency, agent.id)
        ]
        return agent.id, agent.respond_phase_a(round_index, neighbor_msgs)

    phase_a_outputs: Dict[int, PhaseAOutput] = _run_parallel(agents, call_phase_a)

    # --- Phase B ---
    def call_phase_b(agent: Agent) -> Tuple[int, PhaseBOutput]:
        neighbor_drafts = [
            (j, phase_a_outputs[j].draft_message)
            for j in neighbors(adjacency, agent.id)
        ]
        return agent.id, agent.respond_phase_b(round_index, question_prompt, neighbor_drafts)

    phase_b_outputs: Dict[int, PhaseBOutput] = _run_parallel(agents, call_phase_b)

    return RoundEntry(
        round=round_index,
        phase_a=[PhaseAEntry(id=a.id, draft_message=phase_a_outputs[a.id].draft_message) for a in agents],
        phase_b=_to_phase_b_entries(agents, phase_b_outputs),
    )


def _run_parallel(agents: List[Agent], fn) -> Dict[int, object]:
    """Run fn(agent) for all agents in parallel, return {agent_id: result}."""
    results: Dict[int, object] = {}
    with ThreadPoolExecutor(max_workers=len(agents)) as executor:
        futures = {executor.submit(fn, agent): agent for agent in agents}
        for future in as_completed(futures):
            agent_id, output = future.result()
            results[agent_id] = output
    return results


def _to_phase_b_entries(agents: List[Agent], outputs: Dict[int, PhaseBOutput]) -> List[PhaseBEntry]:
    return [
        PhaseBEntry(
            id=a.id,
            belief=outputs[a.id].belief,
            belief_reasoning=outputs[a.id].belief_reasoning,
            public_message=outputs[a.id].public_message,
        )
        for a in agents
    ]
