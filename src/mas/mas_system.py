import random
import uuid
from typing import Callable, Dict, List, Optional

from langchain_core.language_models import BaseChatModel

from src.mas.agent import Agent
from src.mas.logging import AgentMeta, RoundEntry, RunResult
from src.mas.round_runner import run_round
from src.mas.topology import fully_connected


class MultiAgentSystem:
    def __init__(
        self,
        n: int,
        t: int,
        llm: BaseChatModel,
        w: Optional[int] = 1,
        adjacency: Optional[List[List[int]]] = None,
        verbose: bool = False,
    ) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if t < 1:
            raise ValueError(f"t must be >= 1, got {t}")
        if w is not None and w < 1:
            raise ValueError(f"w must be >= 1 or None (infinite), got {w}")

        self._n = n
        self._t = t
        self._w = w
        self._verbose = verbose
        self._llm = llm
        self._adjacency: List[List[int]] = (
            adjacency if adjacency is not None else fully_connected(n)
        )

    def run(
        self,
        question: str,
        options: Dict[str, str],
        question_id: str,
        ground_truth: str,
        question_prompts: List[str],
        on_round_complete: Optional[Callable[[RoundEntry], None]] = None,
    ) -> RunResult:
        names = [f"Agent{i + 1}" for i in range(self._n)]
        random.shuffle(names)
        agents = [
            Agent(agent_id=i, name=names[i], llm=self._llm, w=self._w, verbose=self._verbose)
            for i in range(self._n)
        ]

        trajectory: List[RoundEntry] = []
        for t in range(self._t + 1):
            round_entry = run_round(
                agents=agents,
                question_prompts=question_prompts,
                adjacency=self._adjacency,
                round_index=t,
                trajectory=trajectory,
                w=self._w,
            )
            trajectory.append(round_entry)
            if on_round_complete is not None:
                on_round_complete(round_entry)

        model_name = _get_model_name(self._llm)
        agent_metas = [
            AgentMeta(id=a.id, model=model_name, persona=a.persona) for a in agents
        ]

        return RunResult(
            run_id=str(uuid.uuid4()),
            question_id=question_id,
            question=question,
            options=options,
            ground_truth=ground_truth,
            N=self._n,
            T=self._t,
            W=self._w,
            topology=self._adjacency,
            agents=agent_metas,
            trajectory=trajectory,
        )


def _get_model_name(llm: BaseChatModel) -> str:
    for attr in ("model_name", "model_id", "model"):
        if value := getattr(llm, attr, None):
            return str(value)
    return type(llm).__name__
