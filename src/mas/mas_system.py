import random
import uuid
from typing import Callable, Dict, List, Optional

from langchain_core.language_models import BaseChatModel

from src.mas.agent import Agent
from src.mas.logging import AgentMeta, RoundEntry, RunResult
from src.mas.round_runner import run_round
from src.mas.topology import TOPOLOGY_NAMES, chain, fully_connected, ring, star

_TOPOLOGY_FACTORIES: Dict[str, Callable[[int, int], List[List[int]]]] = {
    "fc": lambda n, hub: fully_connected(n),
    "ring": lambda n, hub: ring(n),
    "chain": lambda n, hub: chain(n),
    "star": lambda n, hub: star(n, hub),
}


class MultiAgentSystem:
    def __init__(
        self,
        n: int,
        t: int,
        llm: BaseChatModel,
        w: Optional[int] = 1,
        topology_name: str = "fc",
        verbose: bool = False,
    ) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if t < 1:
            raise ValueError(f"t must be >= 1, got {t}")
        if w is not None and w < 1:
            raise ValueError(f"w must be >= 1 or None (infinite), got {w}")
        if topology_name not in _TOPOLOGY_FACTORIES:
            raise ValueError(
                f"Unknown topology '{topology_name}'. Available: {TOPOLOGY_NAMES}"
            )

        self._n = n
        self._t = t
        self._w = w
        self._verbose = verbose
        self._llm = llm
        self._topology_name = topology_name
        hub = random.randint(0, n - 1) if topology_name == "star" else 0
        self._adjacency = _TOPOLOGY_FACTORIES[topology_name](n, hub)

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
            topology_name=self._topology_name,
            topology=self._adjacency,
            agents=agent_metas,
            trajectory=trajectory,
        )


def _get_model_name(llm: BaseChatModel) -> str:
    for attr in ("model_name", "model_id", "model"):
        if value := getattr(llm, attr, None):
            return str(value)
    return type(llm).__name__
