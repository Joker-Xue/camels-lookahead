#!/bin/bash
# Full pipeline. Steps 1-3 take minutes to an hour; step 4 trains 47 LSTMs (~7.5 GPU-hours each).
# To check the statistics without retraining, skip step 4 and unpack the archived predictions
# into results/runs (see README), then run steps 5-6.
set -euo pipefail
cd "$(dirname "$0")"
python code/download_camels.py              # 1. CAMELS v1.2 from Zenodo (~7.6 GB, MD5-checked)
python code/build_dataset.py                # 2. per-basin daily arrays in data/processed
python code/validate_attrs.py               # 3a. recomputed vs official climate indices
python code/build_attrs.py                  # 3b. attribute tables for configurations A-G
python code/run_all.py --workers 1          # 4. all training runs (resumable; --workers N for parallel GPU runs)
python code/analyze.py                      # 5. pre-registered analysis -> results/analysis.json
python code/analyze_g.py                    #    exploratory G-F (deviation 2) -> results/analysis_g.json
python code/figures.py                      # 6. figures -> paper/
