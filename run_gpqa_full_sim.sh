#!/usr/bin/env bash
python run_mas.py --dataset gpqa --sample-subset 3 --model mistral-medium --n 4 --t 15 --w 1 2 5 --topology fc --r 50 --workers 5 --early-stopping --u 3 --run-name gpqa_full_sim_run_db --skip-existing
