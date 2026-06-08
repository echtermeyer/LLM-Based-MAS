#!/usr/bin/env bash
python run_mas.py --dataset gpqa --all --model mistral-medium --t 15 --w 1 --topology fc --r 3 --workers 4 --early-stopping --u 3 --subfolder gpqa_subset_scan_nothink --skip-existing
python run_mas.py --dataset hiddenbench --all --model mistral-medium --t 15 --w 1 --topology fc --r 3 --workers 4 --early-stopping --u 3 --subfolder hiddenbench_subset_scan_nothink --skip-existing
