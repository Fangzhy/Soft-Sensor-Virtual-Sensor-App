# Stage 5: feature importance, residuals, and prediction intervals

Generate data and use **Compare models** with **Include Stage 5 diagnostics
and 90% intervals** checked (the default). Compare candidates as before, then
scroll to **Explainability and uncertainty**. No new dependency is required.
Restart the servers if they were running before this update.

## Three distinct data roles

At the default settings, 1,000 rows are divided into:

| Portion | Rows | Purpose |
| --- | --- | --- |
| Proper training | 600 | Five-fold model selection, scaling, and final fitting |
| Calibration | 200 | Compute absolute prediction errors to set interval width |
| Test | 200 | Evaluate predictions, interval coverage, importance, and residuals |

The outer test split stays the same as Stage 4. Then 20% of **all** rows
(rounded up) is removed from the remaining data for calibration. Changing the
test fraction changes the proper-training count; calibration remains 20%.
At the minimum 100 rows there are at least 20 calibration observations and
40 proper-training observations, enough for the fixed five folds.

Every CV fold uses only proper-training rows. The winning model is refitted
on the whole proper-training portion, never on calibration or test data.
Calibration labels cannot influence which model wins. Scores can differ from
Stage 4 because fewer rows are now available for fitting and CV. Uncheck Stage 5
to reproduce the earlier two-way split. The Stage 3 walkthrough remains a
separate two-way split exercise without these diagnostics.

## Split-conformal prediction intervals

For each calibration row, compute `abs(actual − prediction)`. With `m`
calibration rows and desired coverage 90%, calculate:

```text
k = ceil((m + 1) × 0.90)
q = kth smallest absolute calibration error (one-based rank)
interval for a new prediction = [prediction − q, prediction + q]
```

For 200 calibration rows, k = 181. We select that observed error directly,
without interpolating a percentile. For 20 rows, k = 19. The implementation
rejects sample sizes too small for a finite interval at this coverage level.

This basic method has constant width `2q`. It combines model error and
unobserved outcome variability; it does not separately quantify sensor error,
parameter uncertainty, or process drift. We keep bounds unclipped, so they can
extend outside 0–100%. Intervals predict individual outcomes, not the expected
mean concentration or a regression coefficient.

Under exchangeability of calibration and future observations, with the fitted
model independent of calibration labels, split conformal provides a marginal
coverage guarantee. It does **not** guarantee 90% coverage at every sensor
setting, in every concentration range, or on each finite test set. Process
drift, distribution shift, and dependent time-series samples can break these
assumptions. Repeatedly tuning based on coverage also invalidates the intended
separation of roles.

The page reports nominal coverage, observed **test** coverage, and mean width.
The band plot is sorted by predicted concentration, not time. Coverage is
calculated from all held-out rows, with inclusive interval endpoints.

## Permutation feature importance

The model remains frozen. For each sensor, shuffle its test values ten times
and measure the increase in test RMSE relative to unshuffled data. Large
increases indicate greater reliance by this model on that feature. Values are
in percentage points of RMSE, not normalized percentage shares. Negative
importance is possible and is not clipped away.

The error bar is the standard deviation across random shuffles, not a
confidence interval. Correlated features, such as our flow and pressure,
can mask or distort individual importance. Shuffling also breaks their joint
distribution. Importance is model- and dataset-specific, not a causal effect
or evidence of physical mechanism. These test-set diagnostics are for final
inspection, not feature selection or subsequent tuning on the same test set.

## Residual diagnostics

Residual is `actual − predicted`. Positive mean residual indicates average
underprediction. The panel shows bias, population residual SD, a histogram,
residuals versus actual values, and summaries by actual-concentration tertiles.
Repeated tertile boundaries are merged; the final range includes its upper
bound, while earlier ranges exclude theirs.

The correlation between absolute residual and predicted concentration is a
descriptive measure of changing error magnitude. It is undefined when either
quantity is constant. It is not a formal test, nor does a near-zero result
rule out nonlinear patterns. Residuals contain actual values mathematically,
so residual-versus-actual patterns alone cannot diagnose model misspecification.
Inspect the residual-versus-predicted plot and sample counts as well.

## Read the code

1. `backend/schemas.py` adds `include_diagnostics` and typed diagnostic outputs.
2. `backend/services/model_comparison.py` reserves calibration observations
   before CV, freezes the selected model, and calls inspection after fitting.
3. `backend/services/diagnostics.py` implements the finite-sample rank,
   intervals, permutation importance, and residual summaries.
4. `frontend/api_client.py` verifies split counts, finite numbers, interval
   alignment, and reported coverage before plotting.
5. `frontend/model_diagnostics.py` renders the three explanatory tabs.

Use the existing `POST /models/compare` endpoint with
`"include_diagnostics": true`. The API default remains false to preserve
Stage 4 callers; the UI enables it by default. Responses include a
`diagnostics` object when enabled and null otherwise. Results stay associated
with the stored comparison and are cleared on successful data regeneration.
This stage reports intervals on held-out observations; Stage 6 will add new
sensor-input prediction and AI explanations.

## Learning checkpoint

1. Generate 1,000 rows with seed 42 and noise 1. Compare all four models with
   diagnostics enabled. Confirm the 600/200/200 split shown on the page.
2. Inspect permutation importance. Is density the strongest predictor for the
   selected model? Avoid assuming the ranking before looking at the result.
3. Compare low- and high-concentration residual summaries and their counts.
4. Look at observed coverage versus nominal 90%. Why need they not match?
5. Increase noise to 5, regenerate, and compare again. Look for wider intervals,
   but remember finite random samples need not change every statistic monotonically.
6. Explain why using test residuals to set interval width would invalidate this
   test as an independent check of coverage.

Tests: `python -m pytest -q`. They verify the exact conformal rank, disjoint
fitting/calibration/test roles, constant-target behavior, interval arithmetic,
coverage, malformed-response rejection, and UI state reset.

References: [Angelopoulos and Bates: introduction to conformal prediction](https://arxiv.org/abs/2107.07511)
and [scikit-learn permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html).
