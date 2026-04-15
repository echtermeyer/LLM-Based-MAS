import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd

DATASET_PATH = Path(__file__).parents[2] / "dataset" / "gpqa_diamond.csv"


@dataclass
class QuestionSample:
    question: str
    correct_answer: str
    incorrect_answers: List[str]


@dataclass
class ShuffledSample:
    question: str
    options: dict
    correct_option: str


class DataLoader:
    def __init__(self, path: Path = DATASET_PATH):
        self._df = pd.read_csv(path)

    def load_single(self, index: int) -> QuestionSample:
        """Return the QuestionSample at the given 0-based DataFrame index."""
        row = self._df.iloc[index]
        return QuestionSample(
            question=row["Question"],
            correct_answer=row["Correct Answer"],
            incorrect_answers=[
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ],
        )

    def load(self, n: int = 3) -> List[QuestionSample]:
        samples = []
        for _, row in self._df.head(n).iterrows():
            samples.append(QuestionSample(
                question=row["Question"],
                correct_answer=row["Correct Answer"],
                incorrect_answers=[
                    row["Incorrect Answer 1"],
                    row["Incorrect Answer 2"],
                    row["Incorrect Answer 3"],
                ],
            ))
        return samples


def prepare_samples(samples: List[QuestionSample]) -> List[ShuffledSample]:
    shuffled = []
    for sample in samples:
        answers = [sample.correct_answer] + sample.incorrect_answers
        random.shuffle(answers)
        labels = ["A", "B", "C", "D"]
        options = dict(zip(labels, answers))
        correct_option = labels[answers.index(sample.correct_answer)]
        shuffled.append(ShuffledSample(
            question=sample.question,
            options=options,
            correct_option=correct_option,
        ))
    return shuffled
