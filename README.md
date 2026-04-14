# Multi-Dimensional Sleep Quality Model

This project turns your AP Research idea into a runnable pipeline:

- It combines quantitative sleep metrics with qualitative survey responses.
- It converts answers like `very` stressed or `good` morning mood into normalized numeric features.
- It computes a weighted continuous sleep-quality score instead of a binary label.
- It trains an XGBoost model on the quantitative sleep variables to learn a quantitative-only sleep score.
- It appends that learned quantitative score to the standardized combined feature set.
- It trains a NumPy autoencoder to reconstruct the feature set and preserve latent sleep patterns.
- It evaluates the reconstruction-based model with threshold-sweep AUROC, reconstruction error, and Monte Carlo robustness checks.

## Data You Need

Use one row per night or one row per participant-night. The CSV should contain these columns:

- `record_id`
- `total_sleep_hours`
- `sleep_efficiency_pct`
- `interruptions`
- `sleep_onset_latency_min`
- `wake_variability_min`
- `bedtime_variability_min`
- `perceived_stress`
- `morning_mood`
- `daytime_sleepiness`
- `screen_time_before_bed`

The project already includes:

- [config/feature_schema.json](/Users/narensara11/ressleep/config/feature_schema.json)
- [data/real_sleep_data_template.csv](/Users/narensara11/ressleep/data/real_sleep_data_template.csv)
- [scripts/generate_sample_data.py](/Users/narensara11/ressleep/scripts/generate_sample_data.py)
- [docs/qualitative_scoring_framework.md](/Users/narensara11/ressleep/docs/qualitative_scoring_framework.md)

## Qualitative Measures

These text responses are automatically converted into numbers:

- `perceived_stress`: `not_at_all`, `a_little`, `moderately`, `very`, `extremely`
- `morning_mood`: `awful`, `poor`, `okay`, `good`, `excellent`
- `daytime_sleepiness`: `never`, `rarely`, `sometimes`, `often`, `constantly`
- `screen_time_before_bed`: `none`, `brief`, `moderate`, `heavy`, `very_heavy`

Negative factors such as stress and daytime sleepiness are inverted during normalization, so higher processed values always mean better sleep quality.

This makes the qualitative part of the paper explicit: subjective categories are converted into ordinal scores, then normalized into the same 0-1 framework as the quantitative sleep metrics.

## Run It

Create the sample dataset:

```bash
.venv/bin/python scripts/generate_sample_data.py
```

Train the model and generate outputs:

```bash
.venv/bin/python scripts/run_sleep_model.py
```

If you downloaded BRFSS and want a model-ready CSV in this project format:

```bash
.venv/bin/python scripts/preprocess_brfss.py --input /path/to/LLCP2023.XPT --output data/brfss_model_input.csv
```

Then run the model using that processed file:

```bash
.venv/bin/python scripts/run_sleep_model.py --data data/brfss_model_input.csv
```

Outputs are written to `outputs/`:

- `metrics.json`
- `score_alignment.png`
- `quantitative_stage_alignment.png`
- `correlation_heatmaps.png`
- `latent_space.png`
- `training_loss.png`
- `monte_carlo_stability.png`
- `monte_carlo_distributions.png`
- `monte_carlo_samples.json`

## Method Summary

The continuous sleep score is:

```text
sleep_quality = sum(weight_i * normalized_feature_i)
```

The pipeline now has two levels:

1. `XGBoost quantitative stage`
   The quantitative sleep variables are used to train a regressor that learns a quantitative-only sleep score.

2. `Stacked reconstruction stage`
   The learned quantitative score is appended to the standardized combined dataset, which contains the quantitative variables and the converted qualitative variables. The autoencoder learns a latent representation from that stacked feature space.

The reconstructed output is then used to recompute the original weighted sleep score from the sleep features. AUROC is calculated by sweeping across multiple thresholds of the original continuous score and checking how well reconstructed scores rank higher-quality vs. lower-quality sleep.

## Monte Carlo Simulation

The project includes a paper-style Monte Carlo simulation for uncertainty in the qualitative scoring rubric.

- Each qualitative ordinal score is perturbed by up to `+/- 0.5`.
- Quantitative fields can also receive a small amount of noise.
- Feature weights are randomized on each iteration and then renormalized.
- A new continuous sleep score is recomputed for every iteration.
- The code tracks score stability, AUROC stability, and the variability of randomized weights.
