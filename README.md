# camels-lookahead

Code, pre-registration and results for

> Lin, X. (2026). *Does look-ahead in CAMELS climate attributes inflate LSTM streamflow benchmarks? A pre-registered equivalence study.* Manuscript submitted to *Environmental Modelling & Software*.

The CAMELS-US climate attributes (Addor et al., 2017) are computed over water years 1990–2009, which contains the test period of the standard LSTM benchmark split (Kratzert et al., 2019). This repository trains the benchmark LSTM on 531 basins with official, training-period, test-period or no climate attributes, and measures the effect on test performance against a pre-specified equivalence margin of ±0.01 median NSE.

**Main result (benchmark split, 5-seed ensembles):** official minus training-period attributes, median basin ΔNSE = +0.0024 (95% bootstrap CI 0.0011 to 0.0044). Both positive controls failed their pre-registered criterion. See the paper for interpretation.

## Contents

```
PROTOCOL.md                 pre-registered protocol (Chinese) with deviation log
code/
  experiment.py             splits, attribute configurations A–G, hyperparameters
  download_camels.py        fetch CAMELS v1.2 from Zenodo (MD5-checked)
  build_dataset.py          per-basin daily arrays (Daymet forcings, PET, streamflow)
  camels_attrs.py           climate indices following Addor et al. (2017)
  validate_attrs.py         check recomputed indices against the official ones
  build_attrs.py            attribute tables for every split × configuration
  train.py                  one LSTM run (checkpoints every epoch, resumable)
  run_all.py                all 47 runs in protocol order (resumable)
  metrics.py                NSE, KGE, FHV, FLV
  analyze.py                pre-registered analysis -> results/analysis.json
  analyze_g.py              exploratory positive control G − F (deviation 2)
  figures.py                figures of the paper
data/attrs/                 attribute tables used as model inputs (derived from CAMELS, CC BY 4.0)
data/raw/basin_list_531.txt the 531 basins of Kratzert et al. (2019)
results/                    analysis outputs, per-basin ensemble metrics, attribute validation
logs/protocol_freeze*.sha256  file hashes recorded when the protocol was frozen and at each deviation
reproduce.sh                full pipeline
```

Code comments and the protocol are written in Chinese; identifiers, outputs and this README are in English.

## Configurations

| Config. | Climate indices | Runoff signatures | Runs |
|---|---|---|---|
| A | official (WY1990–2009) | – | S1 seeds 1–5, S2 seeds 1–3 |
| B | training period | – | S1 1–5, S2 1–3 |
| C | none | – | S1 1–5, S2 1–3 |
| D | test period | – | S1 1–5, S2 1–3 |
| E | official | official window | S1 1–5 (pre-registered positive control) |
| F | training period | training period | S1 1–5 |
| G | training period | test period | S1 1–5 (exploratory, deviation 2) |

S1: train 1999-10-01 to 2008-09-30, test 1989-10-01 to 1999-09-30 (benchmark). S2: train 1989-10-01 to 1999-09-30, test 1999-10-01 to 2009-09-30.

## Reproducing

```bash
pip install -r requirements.txt
./reproduce.sh
```

`reproduce.sh` lists the steps. Training a run takes about 7.5 h on a single 8 GB NVIDIA GPU when three runs share it. `run_all.py` uses CUDA if available and otherwise Apple MPS. To run `train.py` directly, pass `--device cuda|mps|cpu` (its default is `mps`). Runs resume from the last completed epoch after an interruption, with all random-number-generator states restored.

**Checking the statistics without retraining.** The test-period predictions of all 47 runs (`<split>_<config>_<seed>.npz`, array `pred` of shape basins × days, plus a JSON with hyperparameters and the training log) are archived on Zenodo, https://doi.org/10.5281/zenodo.24109191. Unpack them into `results/runs/`, run steps 1–3 of `reproduce.sh` (needed for the observations), then `python code/analyze.py` and `python code/analyze_g.py`. The outputs should match `results/analysis.json` and `results/analysis_g.json` exactly. `SHA256SUMS` in the archive lists the checksum of every prediction file.

## Pre-registration

`PROTOCOL.md` was frozen before any comparison between configurations was computed. Two deviations are recorded at its end:

- Deviation 1, made before any comparison, restored the benchmark model size.
- Deviation 2, made after the pre-registered results were known, added the exploratory configuration G.

The hash files in `logs/` record the protocol, code and attribute tables at each stage. To check that the current files match the last recorded state:

```bash
grep -v '^#' logs/protocol_freeze_dev2.sha256 | shasum -a 256 -c
```

`code/download_camels.py` and `code/figures.py` were added after that record; neither affects the analysis.

## Data

CAMELS v1.2 (Newman et al., 2015; Addor et al., 2017), Zenodo record 15529996, CC BY 4.0. The derived attribute tables in `data/attrs/` are redistributed under the same licence.

## License

Code: MIT (see `LICENSE`).
