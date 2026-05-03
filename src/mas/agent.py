from typing import List, Literal, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class PhaseAOutput(BaseModel):
    draft: str = Field(
        description="Question, challenge, or partial work to share with neighbors"
    )


class PhaseBOutput(BaseModel):
    vote: Literal["A", "B", "C", "D"] = Field(description="Your current best answer")
    reasoning: str = Field(
        description="Your private chain-of-thought (not shared with others)"
    )
    message: str = Field(
        description="The message you broadcast to your neighbors next round"
    )


_PERSONA_TEMPLATE = """\
You are {name}, part of a collaborative group working on a multiple-choice problem. Your goal is to reach the correct answer together.

Each round (after round 0) has two phases:
  - Phase A: broadcast a draft message to your neighbors. Use it to share your thoughts, ask questions, or respond to what others have said. Do not commit to a final vote in Phase A.
  - Phase B: you see your neighbors' Phase A drafts. Produce your updated vote, reasoning, and a public message for the next round.

Be critical — your neighbors may be wrong, and so may you. Share new information and focus on key facts.\
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
        neighbor_messages: List[Tuple[int, str]],
    ) -> PhaseAOutput:
        return self._step(
            self._llm_a,
            _format_turn(
                f"Round {round_index} Phase A. Neighbor messages from round {round_index - 1}:",
                neighbor_messages,
                "Now produce your Phase A draft.",
            ),
        )

    def respond_phase_b(
        self,
        round_index: int,
        question_prompt: str,
        neighbor_drafts: List[Tuple[int, str]],
    ) -> PhaseBOutput:
        content = (
            question_prompt
            if round_index == 0
            else _format_turn(
                f"Round {round_index} Phase B. Neighbor drafts from round {round_index} Phase A:",
                neighbor_drafts,
                "Now produce your Phase B vote update.",
            )
        )
        return self._step(self._llm_b, content)


def _format_turn(header: str, messages: List[Tuple[int, str]], instruction: str) -> str:
    lines = [header] + [f"- Agent{agent_id + 1}: {msg}" for agent_id, msg in messages]
    lines.append(f"\n{instruction}")
    return "\n".join(lines)
