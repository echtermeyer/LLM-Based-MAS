import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from src.benchmark.dataloader import DataLoader, prepare_samples
from src.mas import MultiAgentSystem
from src.models.llms import Models
from src.utils.console import BOLD, GRAY, GREEN, RED, RESET, print_round

RESULTS_DIR = Path("results/mas")

parser = argparse.ArgumentParser(
    description="Run synchronous-round MAS debate on a GPQA Diamond question."
)
parser.add_argument("--model", choices=Models.NAMES, default="gpt-4o")
parser.add_argument("--n", type=int, default=3, help="Number of agents")
parser.add_argument(
    "--t", type=int, default=5, help="Number of rounds (agents run t=0..T)"
)
parser.add_argument(
    "--temperature", type=float, default=1.0, help="Sampling temperature (default: 1.0)"
)
parser.add_argument(
    "--index", type=int, default=56, help="0-based question index (56 = Q57)"
)
args = parser.parse_args()

sample = DataLoader().load_single(args.index)
shuffled = prepare_samples([sample])[0]

print(f"\n{BOLD}Question index:{RESET} {args.index}")
print(f"{BOLD}Question:{RESET} {shuffled.question}\n")
print(f"{BOLD}Options:{RESET}")
for label, text in shuffled.options.items():
    if label == shuffled.correct_option:
        print(f"  {GREEN}{label}: {text.strip()}{RESET}")
    else:
        print(f"  {label}: {text.strip()}")

print(
    f"\n{BOLD}Running MAS — N={args.n} agents, T={args.t} rounds, model={args.model}, temperature={args.temperature}{RESET}\n"
)
llm = Models.create(args.model, args.temperature)
mas = MultiAgentSystem(n=args.n, t=args.t, llm=llm)

result = mas.run(
    question=shuffled.question,
    options=shuffled.options,
    question_id=str(args.index),
    ground_truth=shuffled.correct_option,
    on_round_complete=lambda r: print_round(r, shuffled.correct_option),
)

final_round = result.trajectory[-1]
vote_counts = Counter(e.belief for e in final_round.phase_b)
majority_answer, _ = vote_counts.most_common(1)[0]
majority_correct = majority_answer == shuffled.correct_option
color = GREEN if majority_correct else RED
mark = "✓" if majority_correct else "✗"

print(f"{BOLD}Final round (t={final_round.round}) beliefs:{RESET} {dict(vote_counts)}")
print(
    f"{BOLD}Majority vote:{RESET} {color}[{mark}] {majority_answer}{RESET}  (correct: {shuffled.correct_option})"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"{timestamp}_{args.model}_N{args.n}_T{args.t}_temp{args.temperature}_q{args.index}.json"
path = RESULTS_DIR / filename
path.write_text(json.dumps(result.to_dict(), indent=2))
print(f"\n{GRAY}Saved → {path}{RESET}")
