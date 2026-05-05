import random
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class PhaseAOutput(BaseModel):
    draft: str = Field(
        description=(
            "Interim communicative act — clarification, agreement, question, "
            "partial argument, or observation. Not a final commitment."
        )
    )


class PhaseBOutput(BaseModel):
    reasoning: str = Field(description="Your private chain-of-thought (never shared)")
    vote: Literal["A", "B", "C", "D"] = Field(description="Your current best answer")
    message: str = Field(description="Public message broadcast to neighbors next round")


@dataclass
class OwnRecord:
    round: int
    vote: str
    reasoning: str
    message: str
    draft: Optional[str]


@dataclass
class PeerRecord:
    name: str
    round: int
    vote: str
    message: str
    draft: Optional[str]


_PERSONA = """\
You are {name}, part of a collaborative group working on a multiple-choice problem. \
Your goal is to reach the correct answer together.

Each round (after round 0) has two phases:
  - Phase A: broadcast a draft message to your neighbors. Share thoughts, ask questions, \
respond to peers. Do not commit to a final answer.
  - Phase B: you see your neighbors' Phase A drafts. Produce your updated vote, \
private reasoning, and public message.

Be critical — your neighbors may be wrong, and so may you.\
"""

_PHASE_A_INST = (
    "Produce an interim communicative act in response to peers' last messages — "
    "clarification, agreement, question, partial argument, or observation. "
    "This is not your final position."
)

_PHASE_B_INST = "Update your belief, private reasoning, and public message."

_ROUND_0_INST = (
    "Carefully read the question and options. "
    "Produce your initial vote, private reasoning, and public message."
)


class Agent:
    def __init__(
        self, agent_id: int, name: str, llm: BaseChatModel, w: Optional[int], verbose: bool = False
    ) -> None:
        self.id = agent_id
        self.name = name
        self.persona = _PERSONA.format(name=name)
        self._llm_a = llm.with_structured_output(PhaseAOutput)
        self._llm_b = llm.with_structured_output(PhaseBOutput)
        self._system = SystemMessage(content=self.persona)
        self._w = w
        self._verbose = verbose
        self._own_history: List[OwnRecord] = []
        self._verbose_buffer: List[str] = []

    def flush_verbose(self) -> None:
        for block in self._verbose_buffer:
            print(block)
        self._verbose_buffer.clear()

    def init_round(self, question_context: str) -> PhaseBOutput:
        content = f"{question_context}\n\n{_ROUND_0_INST}"
        output: PhaseBOutput = self._llm_b.invoke(
            [self._system, HumanMessage(content=content)]
        )
        if self._verbose:
            self._verbose_buffer.append(_print_call(self.name, "Round 0 / Phase B (init)", self.persona, content, output.model_dump()))
        self._own_history.append(
            OwnRecord(
                round=0,
                vote=output.vote,
                reasoning=output.reasoning,
                message=output.message,
                draft=None,
            )
        )
        return output

    def phase_a(
        self, question_context: str, peer_window: List[PeerRecord]
    ) -> PhaseAOutput:
        round_index = len(self._own_history)
        content = (
            _build_context(question_context, self._windowed(), peer_window)
            + f"\n\n{_PHASE_A_INST}"
        )
        output = self._llm_a.invoke([self._system, HumanMessage(content=content)])
        if self._verbose:
            self._verbose_buffer.append(_print_call(self.name, f"Round {round_index} / Phase A", self.persona, content, output.model_dump()))
        return output

    def phase_b(
        self,
        question_context: str,
        own_draft: str,
        peer_window: List[PeerRecord],
        peer_drafts: List[Tuple[str, str]],
    ) -> PhaseBOutput:
        round_index = len(self._own_history)
        ctx = _build_context(question_context, self._windowed(), peer_window)
        drafts_block = (
            f"\n\n--- Your Phase A draft this round ---\n{own_draft}"
            "\n\n--- Peer Phase A drafts this round (randomized order) ---\n"
            + "\n".join(f"{name}: {draft}" for name, draft in peer_drafts)
        )
        content = ctx + drafts_block + f"\n\n{_PHASE_B_INST}"
        output: PhaseBOutput = self._llm_b.invoke(
            [self._system, HumanMessage(content=content)]
        )
        if self._verbose:
            self._verbose_buffer.append(_print_call(self.name, f"Round {round_index} / Phase B", self.persona, content, output.model_dump()))
        self._own_history.append(
            OwnRecord(
                round=len(self._own_history),
                vote=output.vote,
                reasoning=output.reasoning,
                message=output.message,
                draft=own_draft,
            )
        )
        return output

    def _windowed(self) -> List[OwnRecord]:
        if self._w is None:
            return list(self._own_history)
        return self._own_history[-self._w :]


_SEP = "─" * 72
_BOLD = "\033[1m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"


def _print_call(
    agent_name: str,
    label: str,
    system_content: str,
    input_content: str,
    output: Dict[str, str],
) -> str:
    lines = [
        f"\n{_BOLD}{_SEP}{_RESET}",
        f"{_BOLD}{agent_name}  │  {label}{_RESET}",
        f"{_BOLD}{_SEP}{_RESET}",
        f"{_YELLOW}[SYSTEM]{_RESET}\n{system_content}",
        f"\n{_YELLOW}[INPUT]{_RESET}\n{input_content}",
        f"\n{_YELLOW}[OUTPUT]{_RESET}",
    ]
    for k, v in output.items():
        lines.append(f"{_BOLD}{k}:{_RESET} {v}")
    lines.append(f"{_BOLD}{_SEP}{_RESET}\n")
    return "\n".join(lines)


def _build_context(
    question_context: str,
    own_window: List[OwnRecord],
    peer_window: List[PeerRecord],
) -> str:
    if not own_window and not peer_window:
        return question_context

    pb_by_round: Dict[int, List[PeerRecord]] = {}
    for rec in peer_window:
        pb_by_round.setdefault(rec.round, []).append(rec)

    own_by_round = {rec.round: rec for rec in own_window}
    all_rounds = sorted(set(own_by_round) | set(pb_by_round))

    parts = [question_context, "\n=== History ==="]
    for rnd in all_rounds:
        parts.append(f"--- Round {rnd} ---")
        if rnd in own_by_round:
            rec = own_by_round[rnd]
            line = f"You: vote={rec.vote} | reasoning: {rec.reasoning} | message: {rec.message}"
            if rec.draft is not None:
                line += f" | draft: {rec.draft}"
            parts.append(line)
        for rec in pb_by_round.get(rnd, []):
            line = f"{rec.name}: vote={rec.vote} | message: {rec.message}"
            if rec.draft is not None:
                line += f" | draft: {rec.draft}"
            parts.append(line)

    return "\n".join(parts)
