from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class PhaseAEntry:
    id: int
    defense: str
    challenge: str
    question: str


@dataclass
class PhaseBEntry:
    id: int
    vote: str
    reasoning: str
    message: str


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
    W: Optional[int]
    topology_name: str
    topology: List[List[int]]
    agents: List[AgentMeta]
    trajectory: List[RoundEntry]
    early_stopping_u: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            "run_id": self.run_id,
            "question_id": self.question_id,
            "question": self.question,
            "options": self.options,
            "ground_truth": self.ground_truth,
            "N": self.N,
            "T": self.T,
            "W": self.W,
            "early_stopping_u": self.early_stopping_u,
            "topology_name": self.topology_name,
            "topology": self.topology,
            "agents": [
                {"id": a.id, "model": a.model, "persona": a.persona}
                for a in self.agents
            ],
            "trajectory": [
                {
                    "round": r.round,
                    **(
                        {
                            "phase_a": [
                                {
                                    "id": e.id,
                                    "defense": e.defense,
                                    "challenge": e.challenge,
                                    "question": e.question,
                                }
                                for e in r.phase_a
                            ]
                        }
                        if r.phase_a is not None
                        else {}
                    ),
                    "phase_b": [
                        {
                            "id": e.id,
                            "vote": e.vote,
                            "reasoning": e.reasoning,
                            "message": e.message,
                        }
                        for e in r.phase_b
                    ],
                }
                for r in self.trajectory
            ],
        }
