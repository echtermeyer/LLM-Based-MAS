import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

HIDDENBENCH_PATH = Path(__file__).parents[2] / "dataset" / "hiddenbench.json"


@dataclass
class HiddenBenchTask:
    id: int | str
    name: str
    description: str
    shared_info: List[str]
    hidden_info: List[str]
    possible_answers: List[str]
    correct_answer: str


class HiddenBenchLoader:
    def __init__(self, path: Path = HIDDENBENCH_PATH):
        import json
        self._tasks = json.loads(path.read_text())

    def __len__(self) -> int:
        return len(self._tasks)

    def load_single(self, index: int) -> HiddenBenchTask:
        raw = self._tasks[index]
        return HiddenBenchTask(
            id=raw["id"],
            name=raw["name"],
            description=raw["description"],
            shared_info=raw["shared_information"],
            hidden_info=raw["hidden_information"],
            possible_answers=raw["possible_answers"],
            correct_answer=raw["correct_answer"],
        )

    def prepare_task(
        self, task: HiddenBenchTask, n: int
    ) -> Tuple[List[str], Dict[str, str], str]:
        hidden = list(task.hidden_info)
        random.shuffle(hidden)
        agent_hidden: List[List[str]] = [[] for _ in range(n)]
        for idx, fact in enumerate(hidden):
            agent_hidden[idx % n].append(fact)

        labels = ["A", "B", "C", "D"][: len(task.possible_answers)]
        answers = list(task.possible_answers)
        random.shuffle(answers)
        options = dict(zip(labels, answers))
        correct_option = labels[answers.index(task.correct_answer)]

        question_prompts = [
            _format_hiddenbench_prompt(
                task.description,
                _shuffle_facts(task.shared_info + agent_hidden[i]),
                options,
            )
            for i in range(n)
        ]
        return question_prompts, options, correct_option


def _shuffle_facts(facts: List[str]) -> List[str]:
    shuffled = list(facts)
    random.shuffle(shuffled)
    return shuffled


def _format_hiddenbench_prompt(
    description: str, facts: List[str], options: Dict[str, str]
) -> str:
    numbered = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts))
    formatted = "\n".join(f"{k}: {v}" for k, v in options.items())
    return (
        f"Scenario:\n{description}\n\n"
        f"Information:\n{numbered}\n\n"
        f"Options:\n{formatted}\n\n"
        "Reason through the scenario carefully, then select the correct answer."
    )
