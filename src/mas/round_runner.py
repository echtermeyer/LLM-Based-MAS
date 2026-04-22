from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Type

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
    if round_index == 0:
        phase_b_outputs = _run_parallel(
            agents,
            lambda agent: (agent.id, agent.respond_phase_b(0, question_prompt, [])),
        )
        return RoundEntry(
            round=0,
            phase_a=None,
            phase_b=_to_entries(PhaseBEntry, agents, phase_b_outputs),
        )

    phase_a_outputs: Dict[int, PhaseAOutput] = _run_parallel(
        agents,
        lambda agent: (
            agent.id,
            agent.respond_phase_a(
                round_index,
                [
                    (j, prev_phase_b_public_messages[j])
                    for j in neighbors(adjacency, agent.id)
                ],
            ),
        ),
    )

    phase_b_outputs: Dict[int, PhaseBOutput] = _run_parallel(
        agents,
        lambda agent: (
            agent.id,
            agent.respond_phase_b(
                round_index,
                question_prompt,
                [
                    (j, phase_a_outputs[j].draft_message)
                    for j in neighbors(adjacency, agent.id)
                ],
            ),
        ),
    )

    return RoundEntry(
        round=round_index,
        phase_a=_to_entries(PhaseAEntry, agents, phase_a_outputs),
        phase_b=_to_entries(PhaseBEntry, agents, phase_b_outputs),
    )


def _run_parallel(agents: List[Agent], fn) -> Dict[int, object]:
    results: Dict[int, object] = {}
    with ThreadPoolExecutor(max_workers=len(agents)) as executor:
        futures = {executor.submit(fn, agent): agent for agent in agents}
        for future in as_completed(futures):
            agent_id, output = future.result()
            results[agent_id] = output
    return results


def _to_entries(cls: Type, agents: List[Agent], outputs: Dict) -> list:
    return [cls(id=a.id, **outputs[a.id].model_dump()) for a in agents]
