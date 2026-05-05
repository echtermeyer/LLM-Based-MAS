from typing import Any, Dict, List

from src.benchmark.benchmarker import BenchmarkResult
from src.mas.logging import RoundEntry

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
GRAY = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"

_SEP = "─" * 72


def print_verbose_call(
    agent_name: str,
    phase: str,
    round_index: int,
    system_content: str,
    input_content: str,
    output: Dict[str, Any],
) -> None:
    print(f"\n{BOLD}{_SEP}{RESET}")
    print(f"{BOLD}{agent_name}  │  Round {round_index}  │  {phase}{RESET}")
    print(f"{BOLD}{_SEP}{RESET}")
    print(f"{YELLOW}[SYSTEM]{RESET}")
    print(system_content)
    print(f"\n{YELLOW}[INPUT]{RESET}")
    print(input_content)
    print(f"\n{YELLOW}[OUTPUT]{RESET}")
    for k, v in output.items():
        print(f"{BOLD}{k}:{RESET} {v}")
    print(f"{BOLD}{_SEP}{RESET}\n")


def print_round(round_entry: RoundEntry, correct_option: str, verbose: bool = False) -> None:
    print(f"{BOLD}Round t={round_entry.round}{RESET}")
    if round_entry.phase_a is not None:
        print(f"  {YELLOW}Phase A:{RESET}")
        for entry in round_entry.phase_a:
            msg = entry.draft
            if verbose:
                print(f"    Agent{entry.id + 1}:\n      draft: {GRAY}{msg}{RESET}")
            else:
                print(
                    f"    Agent{entry.id + 1}: {GRAY}{msg[:80]}{'…' if len(msg) > 80 else ''}{RESET}"
                )
    print(f"  {YELLOW}Phase B:{RESET}")
    for entry in round_entry.phase_b:
        correct = entry.vote == correct_option
        color = GREEN if correct else RED
        if verbose:
            print(
                f"    Agent{entry.id + 1}: vote={color}{entry.vote}{RESET}\n"
                f"      reasoning: {GRAY}{entry.reasoning}{RESET}\n"
                f"      message: {GRAY}{entry.message}{RESET}"
            )
        else:
            print(
                f"    Agent{entry.id + 1}: vote={color}{entry.vote}{RESET}  "
                f"| message={GRAY}{entry.message[:80]}{'…' if len(entry.message) > 80 else ''}{RESET}"
            )
    print()

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
