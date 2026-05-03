from dataclasses import dataclass
from typing import List, Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel
from tqdm import tqdm

from src.benchmark.gpqa_loader import QuestionSample, ShuffledSample


class MCAnswer(BaseModel):
    reasoning: str
    answer: Literal["A", "B", "C", "D"]


@dataclass
class BenchmarkResult:
    question: str
    options: dict
    correct_option: str
    model_answer: str
    reasoning: str
    correct: bool


def _format_prompt(question: str, options: dict) -> str:
    formatted_options = "\n".join(f"{k}: {v}" for k, v in options.items())
    return (
        f"Question:\n{question}\n\n"
        f"Options:\n{formatted_options}\n\n"
        "Reason through the question step by step, then select the correct answer."
    )


class Benchmarker:
    def __init__(self, llm: BaseChatModel):
        self._llm = llm.with_structured_output(MCAnswer)

    def run(
        self, samples: List[ShuffledSample], verbose: bool = False
    ) -> List[BenchmarkResult]:
        results = []
        iterable = samples if verbose else tqdm(samples, desc="Running", unit="q")
        for sample in iterable:
            try:
                response: MCAnswer = self._llm.invoke(
                    _format_prompt(sample.question, sample.options)
                )
                results.append(
                    BenchmarkResult(
                        question=sample.question,
                        options=sample.options,
                        correct_option=sample.correct_option,
                        model_answer=response.answer,
                        reasoning=response.reasoning,
                        correct=response.answer == sample.correct_option,
                    )
                )
            except OutputParserException as e:
                results.append(
                    BenchmarkResult(
                        question=sample.question,
                        options=sample.options,
                        correct_option=sample.correct_option,
                        model_answer="ERROR",
                        reasoning=str(e),
                        correct=False,
                    )
                )
        return results
