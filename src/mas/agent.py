from typing import List, Literal, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class PhaseAOutput(BaseModel):
    draft_message: str = Field(
        description="Question, challenge, or partial work to share with neighbors"
    )


class PhaseBOutput(BaseModel):
    belief: Literal["A", "B", "C", "D"] = Field(description="Your current best answer")
    belief_reasoning: str = Field(
        description="Your private chain-of-thought (not shared with others)"
    )
    public_message: str = Field(
        description="The message you broadcast to your neighbors next round"
    )


_PERSONA_TEMPLATE = """\
You are {name}, part of a collaborative group solving a multiple-choice question. Work with your fellow agents to find the correct answer.

Each round (after round 0) has two phases:
  - Phase A: broadcast a draft message. Use it to ask clarifying questions, challenge a neighbor's reasoning, or share partial work. Do not commit to a belief in Phase A.
  - Phase B: you see your neighbors' Phase A drafts. Answer any questions directed at you, then produce your updated belief, reasoning, and public message.
You must be very critical with other agents reasoning as they might be incorrect. Furthermore, you might be incorrect as well.\
"""


class Agent:
    def __init__(self, agent_id: int, llm: BaseChatModel) -> None:
        self.id = agent_id
        self.name = f"Agent{agent_id + 1}"
        self.persona = _PERSONA_TEMPLATE.format(name=self.name)
        self._llm_a = llm.with_structured_output(PhaseAOutput)
        self._llm_b = llm.with_structured_output(PhaseBOutput)
        self._system = SystemMessage(content=self.persona)
        self._history: List[BaseMessage] = []

    def _step(self, llm, content: str):
        self._history.append(HumanMessage(content=content))
        output = llm.invoke([self._system] + self._history)
        self._history.append(AIMessage(content=output.model_dump_json()))
        return output

    def respond_phase_a(
        self,
        round_index: int,
        neighbor_public_messages: List[Tuple[int, str]],
    ) -> PhaseAOutput:
        return self._step(
            self._llm_a,
            _format_turn(
                f"Round {round_index} Phase A. Neighbor public messages from round {round_index - 1}:",
                neighbor_public_messages,
                "Now produce your Phase A draft message.",
            ),
        )

    def respond_phase_b(
        self,
        round_index: int,
        question_prompt: str,
        neighbor_draft_messages: List[Tuple[int, str]],
    ) -> PhaseBOutput:
        content = (
            question_prompt
            if round_index == 0
            else _format_turn(
                f"Round {round_index} Phase B. Neighbor drafts from round {round_index} Phase A:",
                neighbor_draft_messages,
                "Now produce your Phase B belief update.",
            )
        )
        return self._step(self._llm_b, content)


def _format_turn(header: str, messages: List[Tuple[int, str]], instruction: str) -> str:
    lines = [header] + [f"- Agent{agent_id + 1}: {msg}" for agent_id, msg in messages]
    lines.append(f"\n{instruction}")
    return "\n".join(lines)
