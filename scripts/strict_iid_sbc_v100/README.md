# Strict IID SBC for TimeFlow v1_a (v100)

This directory contains a separate, reproducible strict-IID simulation-based
calibration pipeline. It does not modify the existing training or held-out test
data.

## Statistical design

For every population `i` the pipeline independently:

1. draws the eight inferred hyperparameters from their exact uniform priors;
2. simulates a new CBC population using a seed never used by the training set;
3. generates a new H1/L1 signal realization with the v100 no-double-redshift
   convention and removes individual events with network SNR >= 12;
4. generates fresh H1/L1 O4 noise and adds it with the training scale `1/100`;
5. saves exactly one H1/L1 observation for that hyperparameter vector;
6. draws 999 TimeFlow posterior samples and computes the Stan SBC rank.

The default configuration conditions on a non-empty CBC population, matching
the training generator's non-empty requirement. A retry changes only the event
random seed; it never changes the sampled hyperparameters. The resulting target
is therefore `p(d | theta, N_event > 0)`. Set
`CONDITION_ON_NONEMPTY_POPULATION = False` only if the intended model includes
zero-event/pure-noise frames.

## Files

- `config.py`: all paths, physical settings, sample counts and disjoint seeds.
- `generate_iid_hyperparameters.py`: IID uniform parameter generation (no LHS).
- `generate_iid_dataset.py`: CBC population, waveform, O4 noise and final NPY generation.
- `evaluate_iid_sbc.py`: resumable posterior-rank evaluation and 800-dpi PDF.
- `run_pipeline.py`: sequentially runs all three stages.

## Recommended execution

Run from `/workspace/NN_SGWB/scripts` in the same Conda environment used for
training:

```bash
conda activate LIGO_SGWB
python strict_iid_sbc_v100/generate_iid_hyperparameters.py
python strict_iid_sbc_v100/generate_iid_dataset.py
python strict_iid_sbc_v100/evaluate_iid_sbc.py
```

Or run the complete pipeline:

```bash
python strict_iid_sbc_v100/run_pipeline.py
```

`run_pipeline.py` automatically restarts the data-generation stage after an
occasional native-library crash such as `SIGFPE`. Completed H1/L1 pairs are
kept and skipped on restart. After 21 consecutive failures it stops and reports
that the same population may be causing a persistent failure.

Generation is restart-safe: an observation is skipped only when both its H1
and L1 files exist. To distribute generation manually, use non-overlapping
half-open ranges:

```bash
python strict_iid_sbc_v100/generate_iid_dataset.py --start 0 --stop 1000
python strict_iid_sbc_v100/generate_iid_dataset.py --start 1000 --stop 2000
```

Do not run overlapping ranges concurrently. The final SBC evaluator requires
all `NUM_SBC_SAMPLES` paired records.

## Outputs

All outputs are kept under `strict_iid_sbc_v100/outputs/`:

- `iid_hyperparameters_v100.npz` and `.json` metadata;
- `processed_iid_sbc_v100/H1_splits/` and `L1_splits/`;
- `generation_log.csv` with signal/noise seeds and event counts;
- `strict_iid_sbc_progress_v1_a.npz` for restart-safe evaluation;
- `strict_iid_sbc_metrics_v1_a.txt`;
- `strict_iid_sbc_rank_hist_v1_a.pdf`.

At 512 Hz and 2048 seconds, 1000 two-detector float32 observations require
about 7.8 GiB before filesystem overhead. The waveform stage is also expensive;
start with a small `NUM_SBC_SAMPLES` smoke test if the server environment has
not yet been validated.
