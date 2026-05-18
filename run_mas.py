import argparse
import json
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from src.benchmark.benchmarker import _format_prompt
from src.benchmark.gpqa_loader import GPQALoader, prepare_samples
from src.benchmark.hiddenbench_loader import HiddenBenchLoader
from src.mas import MultiAgentSystem
from src.models.llms import Models
from src.mas.topology import TOPOLOGY_NAMES
from src.utils.console import BOLD, GRAY, GREEN, RED, RESET, print_round

RESULTS_DIR = Path("results/mas")

parser = argparse.ArgumentParser(
    description="Run synchronous-round MAS debate on a single question."
)
parser.add_argument("--dataset", choices=["gpqa", "hiddenbench"], default="gpqa")
parser.add_argument("--model", choices=Models.NAMES, default="mistral-medium")
parser.add_argument(
    "--n",
    type=int,
    default=4,
    help="Agents (gpqa only; derived from task for hiddenbench)",
)
parser.add_argument(
    "--t", type=int, default=10, help="Number of rounds (agents run t=0..T)"
)
parser.add_argument(
    "--w",
    type=int,
    nargs="+",
    default=[1],
    help="Memory window(s) W — rounds of history visible per call (>= 1). Multiple values run all combos.",
)
parser.add_argument(
    "--topology",
    choices=TOPOLOGY_NAMES,
    nargs="+",
    default=["fc"],
    help="Topology name(s). Multiple values run all combos. fc=fully connected, ring=directed limit-cycle, chain=undirected line, star=hub+leaves (hub randomized per run).",
)
parser.add_argument(
    "--index",
    type=int,
    nargs="+",
    default=[56],
    help="0-based question/task index (multiple values run each in sequence)",
)
parser.add_argument(
    "--r", type=int, default=1, help="Number of independent repetitions"
)
parser.add_argument(
    "--verbose", action="store_true", help="Print full prompts and responses"
)
args = parser.parse_args()

for w_val in args.w:
    if w_val < 1:
        parser.error(f"--w values must be >= 1, got {w_val}")

temperature = Models.TEMPERATURES[args.model]
llm = Models.create(args.model)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

total_combos = len(args.index) * len(args.w) * len(args.topology)
combo_idx = 0

for index in args.index:
    if args.dataset == "hiddenbench":
        loader = HiddenBenchLoader()
        task = loader.load_single(index)
        n = len(task.hidden_info)
        question_prompts, options, correct_option = loader.prepare_task(task, n)
        question = task.description
        question_id = str(index)
    else:
        n = args.n
        sample = GPQALoader().load_single(index)
        shuffled = prepare_samples([sample])[0]
        question_prompts = [_format_prompt(shuffled.question, shuffled.options)] * n
        question = shuffled.question
        options = shuffled.options
        correct_option = shuffled.correct_option
        question_id = str(index)

    for w in args.w:
        for topo in args.topology:
            combo_idx += 1
            print(
                f"\nConfig {combo_idx}/{total_combos}  "
                f"q={index} | W={w} | topo={topo} | N={n} T={args.t} R={args.r} | model={args.model}\n"
                f"  {question[:120].replace(chr(10), ' ')}…"
            )

            repetitions = []
            combo_start = time.monotonic()
            started_at = datetime.now(timezone.utc).isoformat()

            bar = tqdm(range(args.r), unit="rep", leave=True, disable=args.verbose)
            for rep in bar:
                if args.verbose:
                    print(f"\n{BOLD}--- Repetition {rep + 1}/{args.r} ---{RESET}\n")

                rep_start = time.monotonic()
                seed = random.getrandbits(32)
                random.seed(seed)
                mas = MultiAgentSystem(
                    n=n,
                    t=args.t,
                    llm=llm,
                    w=w,
                    topology_name=topo,
                    verbose=args.verbose,
                )
                result = mas.run(
                    question=question,
                    options=options,
                    question_id=question_id,
                    ground_truth=correct_option,
                    question_prompts=question_prompts,
                    on_round_complete=lambda r: (
                        print_round(r, correct_option, verbose=True)
                        if args.verbose
                        else None
                    ),
                )

                final_round = result.trajectory[-1]
                init_round = result.trajectory[0]
                vote_counts = Counter(e.vote for e in final_round.phase_b)
                init_counts = Counter(e.vote for e in init_round.phase_b)
                majority_answer, _ = vote_counts.most_common(1)[0]
                majority_correct = majority_answer == correct_option
                mark = f"{GREEN}✓{RESET}" if majority_correct else f"{RED}✗{RESET}"
                votes_str = " ".join(f"{k}:{v}" for k, v in sorted(vote_counts.items()))
                init_str = " ".join(f"{k}:{v}" for k, v in sorted(init_counts.items()))

                if args.verbose:
                    print(
                        f"\n{BOLD}Majority vote:{RESET} {mark} {majority_answer}  [{votes_str}]  (correct: {correct_option})"
                    )
                else:
                    tqdm.write(
                        f"  rep {rep + 1:>3}: {mark} {majority_answer}  t0=[{init_str}] → tf=[{votes_str}]"
                    )

                rep_dict = result.to_dict()
                rep_dict["repetition"] = rep
                rep_dict["random_seed"] = seed
                rep_dict["majority_answer"] = majority_answer
                rep_dict["correct"] = majority_correct
                rep_dict["duration_s"] = round(time.monotonic() - rep_start, 2)
                repetitions.append(rep_dict)

            n_correct = sum(r["correct"] for r in repetitions)
            print(f"  {n_correct}/{args.r} correct")

            first = repetitions[0]
            output = {
                "started_at": started_at,
                "dataset": args.dataset,
                "question_id": first["question_id"],
                "question": first["question"],
                "options": first["options"],
                "ground_truth": first["ground_truth"],
                "N": first["N"],
                "T": first["T"],
                "W": w,
                "topology_name": topo,
                "temperature": temperature,
                "model": args.model,
                "R": args.r,
                "total_duration_s": round(time.monotonic() - combo_start, 2),
                "repetitions": repetitions,
            }

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = (
                f"{timestamp}_{args.dataset}_{args.model}_N{n}_T{args.t}"
                f"_W{w}_topo{topo}_temp{temperature}_q{index}_R{args.r}.json"
            )
            path = RESULTS_DIR / filename
            path.write_text(json.dumps(output, indent=2))
            print(f"  {GRAY}Saved → {path}{RESET}")
