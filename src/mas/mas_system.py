import uuid
from typing import Callable, Dict, List, Optional

from langchain_core.language_models import BaseChatModel

from src.benchmark.benchmarker import _format_prompt
from src.mas.agent import Agent
from src.mas.logging import AgentMeta, RoundEntry, RunResult
from src.mas.round_runner import run_round
from src.mas.topology import fully_connected


class MultiAgentSystem:
    """
    Synchronous-round multi-agent debate system.

    At each round t, all N agents update in parallel based on the previous
    round's public messages. belief_reasoning is private and never shared.
    """

    def __init__(
        self,
        n: int,
        t: int,
        llm: BaseChatModel,
        adjacency: Optional[List[List[int]]] = None,
    ) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if t < 1:
            raise ValueError(f"t must be >= 1, got {t}")

        self._n = n
        self._t = t
        self._llm = llm
        self._adjacency: List[List[int]] = adjacency if adjacency is not None else fully_connected(n)
        self._agents: List[Agent] = [Agent(agent_id=i, llm=llm) for i in range(n)]

    def run(
        self,
        question: str,
        options: Dict[str, str],
        question_id: str,
        ground_truth: str,
        on_round_complete: Optional[Callable[[RoundEntry], None]] = None,
    ) -> RunResult:
        """
        Run T+1 rounds (t=0 … t=T) and return the full trajectory as a RunResult.

        Parameters
        ----------
        question:            raw question text
        options:             shuffled option mapping, e.g. {"A": "...", "B": "...", ...}
        question_id:         identifier for logging (e.g. dataset index as string)
        ground_truth:        correct option letter
        on_round_complete:   optional callback invoked immediately after each round finishes,
                             useful for printing live progress
        """
        question_prompt = _format_prompt(question, options)
        trajectory = []
        prev_phase_b_public_messages: Optional[List[str]] = None

        for t in range(self._t + 1):
            round_entry = run_round(
                agents=self._agents,
                question_prompt=question_prompt,
                adjacency=self._adjacency,
                round_index=t,
                prev_phase_b_public_messages=prev_phase_b_public_messages,
            )
            trajectory.append(round_entry)
            prev_phase_b_public_messages = [e.public_message for e in round_entry.phase_b]
            if on_round_complete is not None:
                on_round_complete(round_entry)

        model_name = _get_model_name(self._llm)
        agent_metas = [
            AgentMeta(id=a.id, model=model_name, persona=a.persona)
            for a in self._agents
        ]

        return RunResult(
            run_id=str(uuid.uuid4()),
            question_id=question_id,
            question=question,
            options=options,
            ground_truth=ground_truth,
            N=self._n,
            T=self._t,
            topology=self._adjacency,
            agents=agent_metas,
            trajectory=trajectory,
        )


def _get_model_name(llm: BaseChatModel) -> str:
    """Best-effort extraction of a human-readable model name."""
    for attr in ("model_name", "model_id", "model"):
        if value := getattr(llm, attr, None):
            return str(value)
    return type(llm).__name__
