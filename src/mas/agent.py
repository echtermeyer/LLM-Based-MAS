import random
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.outputs import LLMResult
from pydantic import BaseModel, Field


class PhaseAOutput(BaseModel):
    defense: str = Field(
        description=(
            "One concrete reason your CURRENT vote is correct (≤ 2 sentences). "
            "State a NEW point or a specific computational/conceptual claim. "
            "Do NOT restate or summarize your previous reasoning — peers already have it."
        )
    )
    challenge: str = Field(
        description=(
            "Either: quote or paraphrase ONE specific claim from a peer's MOST RECENT round "
            "that you disagree with, and state why (≤ 2 sentences). "
            "Or: write \"concede: [the claim you accept]\" followed by any remaining specific "
            "numerical/conceptual disagreement (if any). "
            "Do not challenge claims a peer has already retracted. "
            'If no current peer claim is challengeable, write "none" and explain why.'
        )
    )
    question: str = Field(
        description=(
            "One specific question to a named peer about their reasoning (≤ 2 sentences). "
            "Must reference a concrete claim or number — "
            '"How does your sum of 9.6 produce 251 fm without 1/k?" is specific; '
            '"What about units?" is not.'
        )
    )


class PhaseBOutput(BaseModel):
    reasoning: str = Field(description="Your private chain-of-thought (never shared)")
    vote: Literal["A", "B", "C", "D"] = Field(description="Your current best answer")
    confidence: int = Field(
        description=(
            "Your private confidence in your current vote as an integer from 0 (completely uncertain) "
            "to 100 (completely certain). Never shared with peers."
        )
    )
    message: str = Field(
        description=(
            "Begin by addressing any questions peers directed at you in this round's Phase A. "
            "For each such question, provide a specific number or short computation if a number "
            "was asked for, or explicitly state \"I cannot produce this value because X.\" "
            "Do not substitute peer consensus for an answer; do not paraphrase the question back. "
            "After answering, state your conclusion as usual."
        )
    )


@dataclass
class OwnRecord:
    round: int
    vote: str
    confidence: int
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
Your goal is to help the group find the correct answer — agreement should follow from better arguments, not the other way around.

Each round (after round 0) has two phases:
  - Phase A: output three structured fields — defense (a new concrete reason your current \
vote is correct), challenge (dispute a specific current peer claim, or concede it and name \
any remaining disagreement), question (a targeted question to a named peer referencing a \
concrete claim or number). Do not change your vote in this phase.
  - Phase B: you see your neighbors' Phase A outputs. Produce your updated vote, \
private reasoning, and public message.

Be critical — your neighbors may be wrong, and so may you.\
"""

_PHASE_A_INST = "Produce your Phase A structured output. Do not change your vote in this phase."

_PHASE_B_INST = (
    "Update your belief, private reasoning, and public message. "
    "If you change your vote, cite the specific Phase A draft (own or peer) that caused the update."
)

_ROUND_0_INST = (
    "Carefully read the question and options. "
    "Produce your initial vote, private reasoning, and public message."
)


class _UsageCapture(BaseCallbackHandler):
    def __init__(self):
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self._any = False

    def on_llm_end(self, response: LLMResult, **kwargs):
        usage = (response.llm_output or {}).get("token_usage", {})
        if usage:
            self.prompt_tokens += usage.get("prompt_tokens") or 0
            self.completion_tokens += usage.get("completion_tokens") or 0
            self._any = True

    def as_dict(self) -> Dict[str, Optional[int]]:
        if not self._any:
            return {"prompt_tokens": None, "completion_tokens": None}
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens}


class Agent:
    def __init__(
        self, agent_id: int, name: str, llm: BaseChatModel, w: Optional[int], verbose: bool = False
    ) -> None:
        self.id = agent_id
        self.name = name
        self.persona = _PERSONA.format(name=name)
        self._llm_a = llm.bind(model_kwargs={"reasoning_effort": "none"}).with_structured_output(PhaseAOutput)
        self._llm_b = llm.bind(model_kwargs={"reasoning_effort": "none"}).with_structured_output(PhaseBOutput)
        self._system = SystemMessage(content=self.persona)
        self._w = w
        self._verbose = verbose
        self._own_history: List[OwnRecord] = []
        self._verbose_buffer: List[str] = []

    def flush_verbose(self) -> None:
        for block in self._verbose_buffer:
            print(block)
        self._verbose_buffer.clear()

    def init_round(self, question_context: str) -> Tuple[PhaseBOutput, Dict[str, Optional[int]]]:
        content = f"{question_context}\n\n{_ROUND_0_INST}"
        cb = _UsageCapture()
        output: PhaseBOutput = self._llm_b.invoke(
            [self._system, HumanMessage(content=content)], config={"callbacks": [cb]}
        )
        if self._verbose:
            self._verbose_buffer.append(_print_call(self.name, "Round 0 / Phase B (init)", self.persona, content, output.model_dump()))
        self._own_history.append(
            OwnRecord(
                round=0,
                vote=output.vote,
                confidence=output.confidence,
                reasoning=output.reasoning,
                message=output.message,
                draft=None,
            )
        )
        return output, cb.as_dict()

    def phase_a(
        self, question_context: str, peer_window: List[PeerRecord]
    ) -> Tuple[PhaseAOutput, Dict[str, Optional[int]]]:
        round_index = len(self._own_history)
        content = (
            _build_context(question_context, self._windowed(), peer_window)
            + f"\n\n{_PHASE_A_INST}"
        )
        cb = _UsageCapture()
        output = self._llm_a.invoke([self._system, HumanMessage(content=content)], config={"callbacks": [cb]})
        if self._verbose:
            self._verbose_buffer.append(_print_call(self.name, f"Round {round_index} / Phase A", self.persona, content, output.model_dump()))
        return output, cb.as_dict()

    def phase_b(
        self,
        question_context: str,
        own_draft: str,
        peer_window: List[PeerRecord],
        peer_drafts: List[Tuple[str, str]],
    ) -> Tuple[PhaseBOutput, Dict[str, Optional[int]]]:
        round_index = len(self._own_history)
        ctx = _build_context(question_context, self._windowed(), peer_window)
        drafts_block = (
            f"\n\n--- Your Phase A draft this round ---\n{own_draft}"
            "\n\n--- Peer Phase A drafts this round (randomized order) ---\n"
            + "\n".join(f"{name}: {draft}" for name, draft in peer_drafts)
        )
        content = ctx + drafts_block + f"\n\n{_PHASE_B_INST}"
        cb = _UsageCapture()
        output: PhaseBOutput = self._llm_b.invoke(
            [self._system, HumanMessage(content=content)], config={"callbacks": [cb]}
        )
        if self._verbose:
            self._verbose_buffer.append(_print_call(self.name, f"Round {round_index} / Phase B", self.persona, content, output.model_dump()))
        self._own_history.append(
            OwnRecord(
                round=len(self._own_history),
                vote=output.vote,
                confidence=output.confidence,
                reasoning=output.reasoning,
                message=output.message,
                draft=own_draft,
            )
        )
        return output, cb.as_dict()

    def _windowed(self) -> List[OwnRecord]:
        if self._w is None:
            return list(self._own_history)
        return self._own_history[-self._w :]


_SEP = "─" * 72


def _format_phase_a(output: PhaseAOutput) -> str:
    return (
        f"defense: {output.defense}\n"
        f"challenge: {output.challenge}\n"
        f"question: {output.question}"
    )
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
            line = f"You: vote={rec.vote} | confidence={rec.confidence} | reasoning: {rec.reasoning} | message: {rec.message}"
            if rec.draft is not None:
                line += f" | draft: {rec.draft}"
            parts.append(line)
        for rec in pb_by_round.get(rnd, []):
            line = f"{rec.name}: message: {rec.message}"
            if rec.draft is not None:
                line += f" | draft: {rec.draft}"
            parts.append(line)

    return "\n".join(parts)
