# Stage 4: comparing model families fairly

Generate a dataset, choose **Compare models**, select candidates, and click
**Compare selected models**. Start with the default 1,000 rows. The app compares
fixed teaching presets, selects the lowest mean cross-validation RMSE, and
then displays held-out test diagnostics for that model.

## Setup

XGBoost is the new dependency. With the project virtual environment active:

```powershell
python -m pip install -r requirements-dev.txt
```

Restart the two servers using the README commands. The **Linear Regression
walkthrough** workflow preserves the Stage 3 single-model exercise.

## Four families, four different approaches

| Model | How it predicts | Fixed preset |
| --- | --- | --- |
| Linear Regression | Adds weighted sensor measurements | StandardScaler + ordinary least squares |
| Random Forest | Averages many decision trees trained on bootstrap samples | 100 trees; depth 10; minimum leaf size 2 |
| XGBoost | Adds trees sequentially to reduce remaining errors | 150 trees; depth 3; learning rate 0.05; histogram tree method |
| Neural Network | Learns nonlinear transformations through hidden layers | 32 and 16 neurons; tanh; L-BFGS; alpha 0.1; tolerance 0.001; 800 iterations maximum |

These are **hyperparameters**: choices made before fitting, unlike learned
coefficients or weights. They keep the demo understandable and reasonably
fast; they are not optimal settings for every dataset. Stage 4 compares
presets and does not perform an automatic hyperparameter search.

Trees use the original sensor units. Linear Regression and the neural network
scale inputs. The neural network also standardizes its target through
`TransformedTargetRegressor`, which automatically converts predictions back
to concentration units. Every scaler learns exclusively from the rows passed
to that particular fit. The target is never an input feature.

## Why five-fold cross-validation?

With 1,000 rows and a 20% test fraction:

```text
1,000 observations
├── 800 training rows → five shared folds
│   Each fit: 640 rows for fitting, 160 rows for validation
│   Repeat five times so each training row is validated once
│   Compare mean validation RMSE across model families
└── 200 held-out test rows → used only after selecting a model
```

The backend creates one train/test split and one shuffled list of five folds.
Every candidate gets those same folds and a fresh estimator for each fit.
Scaling happens inside the estimator, so validation observations cannot affect
scaling parameters. No outer-test rows reach these fold fits.

After comparison, the selected model is freshly fitted on all 800 training
rows, then evaluated on the 200 test rows. Four candidates require 21 fits:
four times five CV fits, followed by one final fit. Seeds control data
partitioning and model randomness. Results are repeatable within the same
software environment; changing versions can change numerical results.

## Reading the comparison

- **CV RMSE mean:** lower is better; this is the selection metric.
- **CV RMSE SD:** population standard deviation of the five fold scores.
  It describes variability across folds, not a prediction interval or a
  confidence interval for generalization performance.
- **CV MAE and R²:** additional perspectives on validation performance.
  Expand the fold table to see every score. If any fold has constant target
  values, aggregate R² is undefined instead of averaging only the other folds.
- **Fold chart:** compare models on each shared validation fold, not only the
  mean. Small ranking differences may reflect sampling variation.
- **Held-out results:** apply only to the CV-selected model, not every candidate.
  They are not used to choose the winner. An exact CV tie uses selection order.

Neural-network training may hit its iteration/function-evaluation limit.
Those results are retained with explicit optimizer notices in the table and
warnings below it. A low error does not imply optimizer convergence; treat a
selected model with notices as a candidate needing further investigation.
No warnings are silently treated as successful convergence.

Comparing repeatedly after inspecting test results can still leak information
through your decisions. Use CV to make choices, not the best-looking test score.
Cross-validation does not turn synthetic-data performance into evidence of
real process performance.

## Read the new code

1. `backend/schemas.py`: comparison request, per-fold scores, and response.
2. `backend/services/model_comparison.py`: model factory, common folds,
   convergence reporting, score aggregation, selection, and final evaluation.
3. `backend/main.py`: `POST /models/compare` connects the service to HTTP.
4. `frontend/api_client.py`: HTTP request and response validation.
5. `frontend/model_comparison.py`: controls, leaderboard, fold chart, and notices.
6. `frontend/model_trainer.py`: shared held-out metrics and plots for any model.

The request includes the displayed dataset's rows, `test_fraction`,
`split_seed`, and a `models` list. Names are `Linear Regression`,
`Random Forest`, `XGBoost`, and `Neural Network`. You can select one candidate
to study its CV performance. Invalid or empty candidate lists receive HTTP 422;
duplicate candidates are evaluated once. Inspect the schema in `/docs`.

The page retains one comparison per session. Chart changes do not retrain.
New form values apply only on Compare. A failed request preserves the previous
comparison with a warning; successful data generation clears all old model
results. The local demo uses a synchronous request (180-second client timeout),
not a background job queue. A client timeout does not cancel an active backend
fit; check its terminal before retrying. Smaller datasets or fewer candidates
reduce runtime. Models are not persisted to disk.

## Learning exercise

1. Generate 1,000 rows with seed 42 and noise 1; compare all four models with
   split seed 42 and a 20% test set. Which model has lowest mean CV RMSE?
2. Inspect its fold variability and optimizer notices before drawing conclusions.
3. Compare the selected model's residual plot with the Stage 3 linear model.
4. Increase target noise to 5 and repeat. Why can no model fully predict the
   random component? Keep settings fixed rather than searching for a lucky seed.
5. Select only Linear Regression. Its holdout predictions should match Stage 3
   for identical data, test fraction, and split seed, despite the extra CV step.

Run `python -m pytest -q` to verify the workflow. Tests cover all four actual
estimators, identical training folds, fresh fits, CV statistics, constant
targets, warnings, API validation, and result invalidation after generation.

Stage 5 will add feature importance, further residual diagnostics, and
prediction intervals with a separate calibration split.

References: [scikit-learn cross-validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html),
[MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html),
and [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html).
