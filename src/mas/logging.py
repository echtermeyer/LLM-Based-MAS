from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class PhaseAEntry:
    id: int
    draft_message: str


@dataclass
class PhaseBEntry:
    id: int
    belief: str
    belief_reasoning: str
    public_message: str


@dataclass
class RoundEntry:
    round: int
    phase_a: Optional[List[PhaseAEntry]]  # None for round 0
    phase_b: List[PhaseBEntry]


@dataclass
class AgentMeta:
    id: int
    model: str
    persona: str


@dataclass
class RunResult:
    run_id: str
    question_id: str
    question: str
    options: Dict[str, str]
    ground_truth: str
    N: int
    T: int
    topology: List[List[int]]
    agents: List[AgentMeta]
    trajectory: List[RoundEntry]

    def to_dict(self) -> Dict:
        return {
            "run_id": self.run_id,
            "question_id": self.question_id,
            "question": self.question,
            "options": self.options,
            "ground_truth": self.ground_truth,
            "N": self.N,
            "T": self.T,
            "topology": self.topology,
            "agents": [{"id": a.id, "model": a.model, "persona": a.persona} for a in self.agents],
            "trajectory": [
                {
                    "round": r.round,
                    **({"phase_a": [{"id": e.id, "draft_message": e.draft_message} for e in r.phase_a]}
                       if r.phase_a is not None else {}),
                    "phase_b": [
                        {
                            "id": e.id,
                            "belief": e.belief,
                            "belief_reasoning": e.belief_reasoning,
                            "public_message": e.public_message,
                        }
                        for e in r.phase_b
                    ],
                }
                for r in self.trajectory
            ],
        }
