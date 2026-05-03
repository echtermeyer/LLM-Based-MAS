import argparse
import json
from datetime import datetime
from pathlib import Path
from src.models.llms import Models
from src.benchmark.gpqa_loader import GPQALoader, prepare_samples
from src.benchmark.benchmarker import Benchmarker, BenchmarkResult
from src.utils.console import BOLD, GRAY, RESET, print_results

RESULTS_DIR = Path("results/benchmarking")

MODEL_CHOICES = {
    "gpt-4o": Models.GPT_4O,
    "claude-sonnet-4": Models.CLAUDE_SONNET_4,
    "claude-sonnet-4.5": Models.CLAUDE_SONNET_45,
    "gemini-pro": Models.GEMINI_PRO,
    "nova-pro": Models.NOVA_PRO,
    "mistral-large": Models.MISTRAL_LARGE,
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=MODEL_CHOICES.keys(), default="gpt-4o")
parser.add_argument("--n", type=int, default=3)
parser.add_argument("--all-models", action="store_true")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

samples = prepare_samples(GPQALoader().load(n=args.n))
models_to_run = (
    MODEL_CHOICES if args.all_models else {args.model: MODEL_CHOICES[args.model]}
)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def save_results(model_name: str, results: list[BenchmarkResult]):
    correct = sum(r.correct for r in results)
    payload = {
        "model": model_name,
        "timestamp": timestamp,
        "score": correct,
        "total": len(results),
        "accuracy": round(correct / len(results), 4),
        "questions": [
            {
                "index": i,
                "question": r.question,
                "option_mapping": r.options,
                "correct_option": r.correct_option,
                "model_answer": r.model_answer,
                "correct": r.correct,
                "reasoning": r.reasoning,
            }
            for i, r in enumerate(results, 1)
        ],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{timestamp}_{model_name}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"{GRAY}Saved → {path}{RESET}")


for model_name, llm in models_to_run.items():
    print(f"\n{BOLD}Running {model_name}...{RESET}")
    results = Benchmarker(llm).run(samples, verbose=args.verbose)
    (
        print_results(model_name, results)
        if args.verbose
        else print(
            f"{BOLD}Score: {sum(r.correct for r in results)}/{len(results)}{RESET}"
        )
    )
    save_results(model_name, results)
