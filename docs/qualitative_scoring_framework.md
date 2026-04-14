# Qualitative-to-Quantitative Scoring Framework

This project converts subjective sleep-related responses into numeric criteria so they can be used in a weighted sleep quality model.

## Core idea

Each qualitative response is mapped onto a 1-5 ordinal scale. The numeric score is then normalized to a 0-1 range before entering the final weighted sleep-quality equation.

```text
normalized_score = (ordinal_score - minimum) / (maximum - minimum)
```

For negative factors such as stress, daytime sleepiness, and screen time, the normalized value is inverted so that larger final values always indicate better sleep quality.

```text
adjusted_score = 1 - normalized_score
```

## Qualitative criteria used

`perceived_stress`

- `1 = not_at_all`
- `2 = a_little`
- `3 = moderately`
- `4 = very`
- `5 = extremely`

`morning_mood`

- `1 = awful`
- `2 = poor`
- `3 = okay`
- `4 = good`
- `5 = excellent`

`daytime_sleepiness`

- `1 = never`
- `2 = rarely`
- `3 = sometimes`
- `4 = often`
- `5 = constantly`

`screen_time_before_bed`

- `1 = none`
- `2 = brief`
- `3 = moderate`
- `4 = heavy`
- `5 = very_heavy`

## Monte Carlo justification

The qualitative rubric is useful, but any assigned score contains uncertainty because self-reported categories are subjective. To account for that uncertainty, the Monte Carlo simulation perturbs each assigned qualitative score by up to `+/- 0.5` during each iteration. Feature weights are also randomized slightly and renormalized. This makes the weighting system probabilistic rather than fixed, which is closer to the methodology described in your paper.

The simulation then recomputes the continuous sleep-quality score and tracks how much the resulting rankings change across thousands of iterations.
