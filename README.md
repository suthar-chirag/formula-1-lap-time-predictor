# GCSRM Task 1 — F1 Lap Time Predictor (Option A)

Predicts lap times within a single F1 race using tire age, and tests whether a
tire-degradation-aware model beats a naive baseline — evaluated with a train/test split
designed to avoid data leakage.

## How to run

Open `F1_Lap_Time_Predictor_Final.ipynb` in Google Colab, upload the four CSV files
below into the same working directory, and run the notebook top to bottom (use
**Runtime → Restart and run all** to make sure it runs cleanly from a fresh session).

Required files (Kaggle: "Formula 1 World Championship 1950–2020" by Rohan Rao):
- `lap_times.csv`
- `pit_stops.csv`
- `results.csv`
- `races.csv`

## Race selection

The notebook uses the **2019 Spanish Grand Prix** — a dry race (sunny, per official
race records) with no red flag.

That race actually had 18 classified finishers, not the 5–10 the task describes. Rather
than discard the race, the notebook narrows the field to the **8 finishers with
multiple pit stops** (so each has at least two stints to work with — one for training,
one held out for testing). This is a deliberate trade-off: it keeps a genuinely dry,
uneventful race with clean data, at the cost of not matching the "5–10 total finishers"
criterion literally. A stricter alternative would have been to search for a race that
itself had 5–10 total finishers (usually due to weather or a first-lap incident) — but
those races more often involve safety cars or partial red flags, which would have
worked against the "dry, no red flag" requirement instead. Given the two conditions
pulled in different directions for this dataset, we prioritized data quality (dry, clean
race) over the literal finisher count.

## Data cleaning

For each of the 8 selected drivers, we remove:
1. **The pit-stop lap itself** — dominated by the pit lane speed limit, not tire wear.
2. **The lap immediately after (out-lap)** — still affected by cold tires and pit exit.
3. **Any remaining lap slower than 1.5× that driver's own median lap time** — catches
   things like being held up behind another car. The median is computed *after* step
   1–2 are removed, so pit-affected laps don't skew it upward.

Exact counts removed at each stage are printed by the notebook (see `Removed pit +
out-laps` / `Removed slow outliers` output).

## Feature engineering

`tire_age`: laps completed on the current set of tires. Resets to 0 on the out-lap
after each pit stop, then increments by 1 every lap until the next stop. `stint`:
which set of tires a lap was run on (1st, 2nd, 3rd...) — used only to drive the
train/test split below, not as a model feature.

## Train/test split — and why it isn't random

Laps next to each other share almost identical conditions (similar fuel load, same
tires, same track state). A random shuffle would put near-identical laps into both the
training and test sets, letting the model effectively "match" a test lap to a
neighboring training lap instead of genuinely learning the tire-wear pattern — this is
**data leakage**, and it would make the reported accuracy look better than it really is.

Instead, for each driver, their **final stint** is held out entirely as test data, and
every earlier stint is used for training. No test lap has a training lap from the same
stint, so the split reflects a genuine "predict laps the model has never seen anything
adjacent to" scenario.

## Models compared

| Feature set | Features |
|---|---|
| Baseline | `grid`, `lap` |
| Enhanced | `grid`, `lap`, `tire_age` |

Each feature set is trained with two algorithms — `LinearRegression` and
`RandomForestRegressor` — and scored on the held-out final stints with RMSE and MAE.

## Results

*(Fill in after running the notebook — pull these straight from the printed
`comparison_table` and `effect_table` outputs.)*

| Feature set | Model | RMSE (ms) | MAE (ms) |
|---|---|---|---|
| Baseline | LinearRegression | _ | _ |
| Baseline | RandomForest | _ | _ |
| Enhanced | LinearRegression | _ | _ |
| Enhanced | RandomForest | _ | _ |

Adding `tire_age` changed RMSE by **_%** (LinearRegression) and **_%** (RandomForest)
relative to baseline. [One or two sentences on what this means — e.g. whether the
enhanced model tracked the rising lap-time trend visible in the stint plot, and whether
one algorithm benefited more than the other.]

## Visualization

`stint_prediction_plot.png` shows one driver's actual lap times against both models'
predictions across their held-out final stint.

## Output files

- `F1_Lap_Time_Predictor_Final.ipynb` — full analysis notebook
- `outputs/model_comparison.csv` — RMSE/MAE for all 4 model combinations
- `outputs/tire_age_effect.csv` — % change in RMSE from adding tire_age, per algorithm
- `outputs/stint_prediction_plot.png` — predicted vs. actual plot for one driver

## Limitations

Results are specific to this one race and driver sample — they show that tire age was
predictive *here*, not that this holds universally across all races or conditions.
Track evolution and fuel-burn effects (cars get lighter, and faster, as the race
progresses) are not modeled separately and may be partly absorbed into the `lap`
feature.
