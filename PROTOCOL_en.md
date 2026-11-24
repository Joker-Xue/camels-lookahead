# Experimental protocol v1 (frozen before any A/B comparison was seen) — English translation

This is a translation of `PROTOCOL.md`, which is the authoritative version and the one whose SHA-256 hashes are recorded in `logs/protocol_freeze*.sha256`. The translation was made on 24 November 2026, after all results of deviations 1 and 2 were known; it adds nothing to the original. Text that was later superseded by a deviation is kept as originally written, as in the Chinese file.

At the time of freezing, only two pilot runs of configuration A (seed 101, not used in the analysis) and summary statistics of the attribute differences had been inspected. All later changes are recorded in the "Deviation log" at the end of this file, and the corresponding analyses are labelled exploratory.

## Data

- CAMELS v1.2 (Zenodo 15529996, CC BY 4.0); 531 basins (the subset of Newman et al. 2017; basin list from kratzert/ealstm_regional_modeling).
- Dynamic inputs: Daymet prcp, srad, tmax, tmin, vp. Target: USGS streamflow, converted to mm/day.
- 18 fixed attributes + 9 climate attributes, the same 27 as in Kratzert et al. (2019).

## Splits

- S1 (benchmark): training 1999-10-01..2008-09-30, test 1989-10-01..1999-09-30.
- S2 (forward): training 1989-10-01..1999-09-30, test 1999-10-01..2009-09-30.
- The official attribute window 1989-10-01..2009-09-30 covers both test periods.

## Configurations and runs

| Config. | Climate attributes | Runoff signatures | S1 seeds | S2 seeds |
|---|---|---|---|---|
| A | official (full window) | none | 1–5 | 1–3 |
| B | recomputed, training period | none | 1–5 | 1–3 |
| C | none | none | 1–5 | 1–3 |
| D | computed from test period | none | 1–5 | 1–3 |
| E | official | full window (q_mean, runoff_ratio, q5, q95, high_q_freq, low_q_freq, zero_q_freq) | 1–5 | not run |
| F | training period | training period (same list) | 1–5 | not run |

42 runs in total. For a given seed, all configurations use the same initialisation seed and sampling seed. Runs are ordered by seed, so that if the experiment is stopped part-way, the completed seeds can still be used as paired comparisons.

## Model (the reduction relative to the benchmark is due to limited compute and must be stated in the paper)

Unidirectional LSTM, 2 layers, 128 hidden units (benchmark: 256), sequence length 270, dropout 0.4, MSE loss, Adam, learning rate 1e-3 / 5e-4 / 1e-4 from epochs 0 / 11 / 21, gradient clipping at 1, batch size 512 (benchmark: 256), 30 epochs, each epoch a random 25% of samples (benchmark: 100%), static attributes concatenated to every time step. Normalisation parameters for dynamic inputs and streamflow from the training period only; static attributes standardised across basins.

## Metrics and estimand

- Per-basin NSE (primary), KGE, FHV, FLV, computed only on days with observations.
- Ensemble: for each configuration, average the daily predictions over the same set of seeds, then compute metrics.

**Primary estimand: under S1, the median across basins of ΔNSE = NSE_A − NSE_B (ensembles).** 95% interval by bootstrap over basins (10,000 replicates, seed 20261102).

**Decision rule (equivalence margin ±0.01 NSE, specified in advance):**
- Interval entirely within (−0.01, +0.01): practically equivalent to current practice.
- Lower bound > +0.01: current practice produces a practically meaningful inflation.
- Upper bound < −0.01: effect in the opposite direction.
- Otherwise: undetermined.

The reasonableness of ±0.01 will be reported against the observed seed-to-seed variation, but the margin will not be changed on that basis.

## Secondary analyses

1. The same estimand under S2 (replication of the primary decision).
2. Paired tests: Wilcoxon signed-rank tests for A vs B, A vs C, D vs B, E vs F, with Holm correction.
3. Positive control: E − F must be clearly positive under S1 (lower interval bound > +0.01). Otherwise "the method cannot detect leakage" and the equivalence conclusion does not hold.
4. Dose–response: Spearman correlation of ΔNSE_(D−B) and ΔNSE_(A−B) with the magnitude of the attribute shift (standardised distance).
5. Groups: report by quartile of the aridity index and by quartile of the relative change in p_mean between test and training periods.
6. The same comparisons for KGE, FHV and FLV.
7. Noise baseline: median absolute ΔNSE between pairs of single models of the same configuration.

## Known limitations (stated in advance)

- The PET coefficients come from the calibration of Newman et al., whose calibration period overlaps the S1 test period. This information could not be removed, so the PET attribute in configuration B is not entirely free of look-ahead.
- Daymet forcings are used instead of the benchmark's Maurer forcings.
- Reduced model size; results are limited to this setting and are not extrapolated to other architectures.
- In simulation mode the test-period forcings are themselves inputs, so look-ahead in climate attributes is not target leakage; the two are examined separately through D/A and E.

## Deviation log

### Deviation 1 (2026-11-03, before any A/B comparison was seen): restore the benchmark model size

- Change: hidden units 128 → 256, batch size 512 → 256, per-epoch sample fraction 25% → 100%, as in Kratzert et al. (2019). All other settings (layers, sequence length, dropout, learning-rate schedule, epochs, clipping, seeds, run order, all statistical analyses) unchanged.
- Reason: compute became available to run at benchmark scale. The statement on reduced size in the "Model" section is void; the limitation "reduced model size" becomes "results are limited to this LSTM setting".
- The 2 runs already completed at v1 size (S1-A-1, S1-B-1) were not evaluated or compared and are not included in the analysis. The seed-101 pilot runs are likewise excluded.
- Training uses float32; several runs may execute at the same time, without sharing any state.

### Deviation 2 (2026-11-09, after all pre-specified results were seen): add a stronger positive control G (exploratory)

- Background: the pre-specified positive control E − F gave +0.0026 [0.0010, 0.0038] under S1, which does not meet "lower bound > +0.01"; by the rule above the equivalence conclusion cannot be drawn. Inspection of the cause: E's runoff signatures are computed over the 20-year full window, only half of which lies in the test period; across basins they correlate with F's (training period) at above 0.9 (mostly > 0.99), and the median standardised difference is only 0.03–0.10, so the leaked information is heavily diluted.
- Change: add configuration G = training-period climate attributes + **runoff signatures computed from the test period only** (the same treatment D applies to climate attributes), S1 only, seeds 1–5, all other settings identical to F. Compare G − F with the same estimand, bootstrap and margin as E − F; the positive criterion is still lower bound > +0.01.
- Status: G was added after the results were known. G − F, and any judgement of sensitivity based on it, is labelled exploratory throughout, reported separately, not included in the pre-specified Holm family, and does not change the computation or output of any pre-specified analysis above. The failure of E − F is reported as is.

### Deviation 3 (2026-11-24, after an internal review; written before any of the new results below were seen): seed uncertainty and static-pathway sensitivity (both exploratory)

Background: the internal review raised two points. (1) The pre-specified interval resamples basins only and does not include training randomness (seeds). (2) Both positive controls failed, so the design's ability to detect a change in the information carried by the static pathway has not been demonstrated directly, and the explanation that "static attributes mainly act as basin identifiers" is supported only indirectly.

Change 1: two-level bootstrap (`code/analyze_seeds.py`, output `results/analysis_seeds.json`).
- In each replicate, first draw seeds with replacement (the same draw for both configurations, to keep the comparison paired) and form the ensembles; then draw basins with replacement and take the median of per-basin ΔNSE. 10,000 replicates, seed 20261102.
- Applied to S1 A−B, A−C, D−B, E−F, G−F and S2 A−B, A−C, D−B. Leave-one-seed-out ensemble estimates are reported as well.
- Reading: if the two-level interval for the primary comparison A−B still lies entirely within ±0.01, the conclusion is robust to seed uncertainty; otherwise this is reported as is and the wording of the conclusion is narrowed accordingly.

Change 2: static-pathway sensitivity controls (S1 only, seeds 1–5, all other settings identical to B).
- H: no static attributes at all (all 27 removed). Compare H − B. Expected to be clearly negative. Criterion: upper bound < −0.01, i.e. "the design can detect a substantial change in the static pathway".
- I: B's 27-dimensional attribute vectors permuted as a whole among basins (a derangement with fixed seed 20261124, so that no basin keeps its own vector). Compare I − B. Reading: interval entirely within ±0.01 supports "attributes mainly act as basin identifiers"; upper bound < −0.01 contradicts it; otherwise undetermined.
- Estimand, ensembling and bootstrap as for A − B (`code/analyze_h.py`, output `results/analysis_h.json`). The attribute tables are derived from `data/attrs/S1_B.csv` by `code/build_attrs_h.py`; existing tables are not modified.

Status: all of the above was added after the pre-specified results were known; it is labelled exploratory throughout, reported separately, not included in the Holm family, and does not change the computation or output of any pre-specified analysis.
