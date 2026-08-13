import logging

logging.getLogger("mlflow").setLevel(logging.ERROR)

import argparse
import json
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from src.benchmark.benchmarker import _format_prompt
from src.benchmark.gpqa_loader import GPQALoader, prepare_samples
from src.mas import MultiAgentSystem, PhaseBEntry, RoundEntry
from src.models.llms import Models
from src.utils.console import BOLD, GRAY, GREEN, RED, RESET, print_round

RESULTS_DIR = Path("results/mas")

W = 2
TOPO = "fc"
N = 4

parser = argparse.ArgumentParser(
    description="Run seeded-init MAS debate: inject round-0 from existing data, debate from round 1."
)
_SCRIPT_DIR = Path(globals().get("__file__", __import__("os").getcwd()))

parser.add_argument(
    "--seed-dir",
    type=Path,
    default=_SCRIPT_DIR / "data/seeds",
    help="Directory containing seed JSON files",
)
parser.add_argument(
    "--results-dir",
    type=Path,
    default=_SCRIPT_DIR / "results/mas",
    help="Root output directory",
)
parser.add_argument(
    "--condition",
    type=int,
    choices=[0, 1],
    nargs="+",
    required=True,
    help="Correct votes in injected round 0 (0 or 1, or both)",
)
parser.add_argument("--model", choices=Models.NAMES, default="mistral-medium")
parser.add_argument("--t", type=int, default=15)
parser.add_argument("--r", type=int, default=50, help="Repetitions per task")
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--early-stopping", action="store_true")
parser.add_argument("--u", type=int, default=3)
parser.add_argument("--skip-existing", action="store_true")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

temperature = Models.TEMPERATURES[args.model]
llm = Models.create(args.model)

seed_files = sorted(args.seed_dir.glob("*gpqa*.json"))
print(f"Found {len(seed_files)} seed files | conditions={args.condition} | R={args.r}")


def _build_initial_round(phase_b_dicts: list) -> RoundEntry:
    entries = [
        PhaseBEntry(
            id=e["id"],
            vote=e["vote"],
            reasoning=e["reasoning"],
            confidence=e["confidence"],
            message=e["message"],
            prompt_tokens=e.get("prompt_tokens"),
            completion_tokens=e.get("completion_tokens"),
        )
        for e in phase_b_dicts
    ]
    return RoundEntry(round=0, phase_a=None, phase_b=entries)


for CONDITION in args.condition:
    output_dir = args.results_dir / f"seeded_init_{CONDITION}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Condition {CONDITION} ===")

    for seed_file in seed_files:
        seed_data = json.loads(seed_file.read_text())
        index = int(seed_data["question_id"])
        gt = seed_data["ground_truth"]

        es_tag = f"_es{args.u}" if args.early_stopping else ""
        pattern = (
            f"*_gpqa_{args.model}_N{N}_T{args.t}"
            f"_W{W}_topo{TOPO}_temp{temperature}_q{index}_R*.json"
        )
        if args.skip_existing and any(output_dir.glob(pattern)):
            print(f"\nSkipping q={index} (already exists)")
            continue

        candidate_rounds = [
            rep["trajectory"][0]["phase_b"]
            for rep in seed_data["repetitions"]
            if sum(e["vote"] == gt for e in rep["trajectory"][0]["phase_b"]) == CONDITION
        ]

        if not candidate_rounds:
            print(f"\nq={index}: no seeds for condition={CONDITION}, skipping")
            continue

        sample = GPQALoader().load_single(index)
        random.seed(index)
        shuffled = prepare_samples([sample])[0]
        random.seed(None)
        question_prompts = [_format_prompt(shuffled.question, shuffled.options)] * N
        question = shuffled.question
        options = shuffled.options
        correct_option = shuffled.correct_option

        rng_seed = random.Random(f"seeded_{CONDITION}_{index}")
        seeds = [rng_seed.getrandbits(32) for _ in range(args.r)]
        sampled_rounds = [
            candidate_rounds[rng_seed.randrange(len(candidate_rounds))]
            for _ in range(args.r)
        ]

        print(
            f"\nq={index} | seeds={len(candidate_rounds)} | "
            f"{question[:100].replace(chr(10), ' ')}…"
        )

        repetitions = []
        combo_start = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        _print_lock = threading.Lock()

        def run_rep(rep: int) -> tuple:
            seed = seeds[rep]
            rng = random.Random(seed)
            rep_start = time.monotonic()
            initial_round = _build_initial_round(sampled_rounds[rep])
            on_complete = (
                (lambda r: print_round(r, correct_option, verbose=True))
                if args.verbose
                else None
            )
            mas = MultiAgentSystem(
                n=N,
                t=args.t,
                llm=llm,
                w=W,
                topology_name=TOPO,
                rng=rng,
                verbose=args.verbose,
                early_stopping_u=args.u if args.early_stopping else None,
            )
            result = mas.run(
                question=question,
                options=options,
                question_id=str(index),
                ground_truth=correct_option,
                question_prompts=question_prompts,
                on_round_complete=on_complete,
                initial_round=initial_round,
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
            rep_dict = result.to_dict()
            rep_dict["repetition"] = rep
            rep_dict["random_seed"] = seed
            rep_dict["majority_answer"] = majority_answer
            rep_dict["correct"] = majority_correct
            rep_dict["duration_s"] = round(time.monotonic() - rep_start, 2)
            line = f"  rep {rep + 1:>3}: {mark} {majority_answer}  t0=[{init_str}] → tf=[{votes_str}]"
            return rep, rep_dict, line

        if args.verbose:
            for rep in range(args.r):
                print(f"\n{BOLD}--- Repetition {rep + 1}/{args.r} ---{RESET}\n")
                _, rep_dict, line = run_rep(rep)
                print(line)
                repetitions.append(rep_dict)
        elif min(args.workers, args.r) <= 1:
            bar = tqdm(range(args.r), unit="rep", leave=True)
            for rep in bar:
                _, rep_dict, line = run_rep(rep)
                tqdm.write(line)
                repetitions.append(rep_dict)
        else:
            workers = min(args.workers, args.r)
            ordered = {}
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(run_rep, rep): rep for rep in range(args.r)}
                with tqdm(total=args.r, unit="rep", leave=True) as bar:
                    for future in as_completed(futures):
                        rep_idx, rep_dict, line = future.result()
                        with _print_lock:
                            tqdm.write(line)
                            bar.update(1)
                        ordered[rep_idx] = rep_dict
            repetitions = [ordered[i] for i in range(args.r)]

        n_correct = sum(r["correct"] for r in repetitions)
        print(f"  {n_correct}/{args.r} correct")

        first = repetitions[0]
        output = {
            "started_at": started_at,
            "dataset": "gpqa",
            "question_id": first["question_id"],
            "question": first["question"],
            "options": first["options"],
            "ground_truth": first["ground_truth"],
            "N": N,
            "T": args.t,
            "W": W,
            "early_stopping_u": args.u if args.early_stopping else None,
            "topology_name": TOPO,
            "temperature": temperature,
            "model": args.model,
            "R": args.r,
            "seeded_condition": CONDITION,
            "n_seed_candidates": len(candidate_rounds),
            "total_duration_s": round(time.monotonic() - combo_start, 2),
            "repetitions": repetitions,
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"{timestamp}_gpqa_{args.model}_N{N}_T{args.t}"
            f"_W{W}_topo{TOPO}_temp{temperature}_q{index}_R{args.r}{es_tag}.json"
        )
        path = output_dir / filename
        path.write_text(json.dumps(output, indent=2))
        print(f"  {GRAY}Saved → {path}{RESET}")
