from typing import List

from src.benchmark.benchmarker import BenchmarkResult
from src.mas.logging import RoundEntry

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
GRAY = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_round(round_entry: RoundEntry, correct_option: str) -> None:
    print(f"{BOLD}Round t={round_entry.round}{RESET}")
    if round_entry.phase_a is not None:
        print(f"  {YELLOW}Phase A:{RESET}")
        for entry in round_entry.phase_a:
            msg = entry.draft_message
            print(
                f"    Agent{entry.id + 1}: {GRAY}{msg[:80]}{'…' if len(msg) > 80 else ''}{RESET}"
            )
    print(f"  {YELLOW}Phase B:{RESET}")
    for entry in round_entry.phase_b:
        correct = entry.belief == correct_option
        color = GREEN if correct else RED
        print(
            f"    Agent{entry.id + 1}: belief={color}{entry.belief}{RESET}  "
            f"| public_message={GRAY}{entry.public_message[:80]}{'…' if len(entry.public_message) > 80 else ''}{RESET}"
        )
    print()


def print_results(model_name: str, results: List[BenchmarkResult]) -> None:
    correct = sum(r.correct for r in results)
    print(f"\n{BOLD}Model: {model_name} — Score: {correct}/{len(results)}{RESET}\n")
    for i, r in enumerate(results, 1):
        if r.correct:
            print(
                f"{GREEN}[✓] Q{i}: expected={r.correct_option}, got={r.model_answer}{RESET}"
            )
        else:
            print(
                f"{RED}[✗] Q{i}: expected={r.correct_option}, got={r.model_answer}{RESET}"
            )
            print(f"{YELLOW}Question:{RESET} {r.question}")
            print(f"{YELLOW}Reasoning:{RESET} {GRAY}{r.reasoning}{RESET}\n")
