from typing import List, Literal, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel


class PhaseAOutput(BaseModel):
    draft_message: str


class PhaseBOutput(BaseModel):
    belief: Literal["A", "B", "C", "D"]
    belief_reasoning: str
    public_message: str


_PERSONA_TEMPLATE = (
    "You are {name}, part of a collaborative group solving a multiple-choice question. "
    "Work with your fellow agents to find the correct answer.\n\n"
    "Each round (after round 0) has two phases:\n"
    "  - Phase A: broadcast a draft message. Use it to ask clarifying questions, challenge a "
    "neighbor's reasoning, or share partial work. Do not commit to a belief in Phase A.\n"
    "  - Phase B: you see your neighbors' Phase A drafts. Answer any questions directed at you, "
    "then produce your updated belief, reasoning, and public message.\n\n"
    'Phase A output schema:  {{"draft_message": "..."}}\n'
    'Phase B output schema:  {{"belief": "<A|B|C|D>", "belief_reasoning": "...", "public_message": "..."}}\n\n'
    "Fields:\n"
    "  - belief: your current best answer (A/B/C/D)\n"
    "  - belief_reasoning: your private chain-of-thought (not shared with others)\n"
    "  - public_message: the message you broadcast to your neighbors next round\n"
    "  - draft_message: Phase A only — question, challenge, or partial work to share"
)


class Agent:
    def __init__(self, agent_id: int, llm: BaseChatModel) -> None:
        self.id = agent_id
        self.name = f"Agent{agent_id + 1}"
        self.persona = _PERSONA_TEMPLATE.format(name=self.name)
        self._llm_a = llm.with_structured_output(PhaseAOutput)
        self._llm_b = llm.with_structured_output(PhaseBOutput)
        self._system = SystemMessage(content=self.persona)
        self._history: List[BaseMessage] = []

    def respond_phase_b(
        self,
        round_index: int,
        question_prompt: str,
        neighbor_draft_messages: List[Tuple[int, str]],
    ) -> PhaseBOutput:
        """
        Phase B (or round-0 initialization): update belief.

        Round 0: history starts with HumanMessage(question).
        Round r≥1: appends the Phase B user turn (neighbor drafts from Phase A).
        """
        if round_index == 0:
            self._history.append(HumanMessage(content=question_prompt))
        else:
            self._history.append(
                HumanMessage(
                    content=_format_phase_b_turn(round_index, neighbor_draft_messages)
                )
            )

        output: PhaseBOutput = self._llm_b.invoke([self._system] + self._history)
        self._history.append(AIMessage(content=output.model_dump_json()))
        return output

    def respond_phase_a(
        self,
        round_index: int,
        neighbor_public_messages: List[Tuple[int, str]],
    ) -> PhaseAOutput:
        """
        Phase A (round r≥1 only): broadcast a draft message.
        Appends the Phase A user turn (neighbors' previous phase-B public messages).
        """
        self._history.append(
            HumanMessage(
                content=_format_phase_a_turn(round_index, neighbor_public_messages)
            )
        )

        output: PhaseAOutput = self._llm_a.invoke([self._system] + self._history)
        self._history.append(AIMessage(content=output.model_dump_json()))
        return output


def _format_phase_a_turn(
    round_index: int, neighbor_public_messages: List[Tuple[int, str]]
) -> str:
    lines = [
        f"Round {round_index} Phase A. Neighbor public messages from round {round_index - 1}:"
    ]
    for agent_id, message in neighbor_public_messages:
        lines.append(f"- Agent{agent_id + 1}: {message}")
    lines.append("\nNow produce your Phase A draft message.")
    return "\n".join(lines)


def _format_phase_b_turn(
    round_index: int, neighbor_draft_messages: List[Tuple[int, str]]
) -> str:
    lines = [
        f"Round {round_index} Phase B. Neighbor drafts from round {round_index} Phase A:"
    ]
    for agent_id, message in neighbor_draft_messages:
        lines.append(f"- Agent{agent_id + 1}: {message}")
    lines.append("\nNow produce your Phase B belief update.")
    return "\n".join(lines)
