# Stage 3: your first predictive model

Generate data, then scroll to **Train and evaluate Linear Regression**. The
default split uses 80% of rows for training and 20% for testing. Press **Train
Linear Regression** to see the results. This stage evaluates predictions for
held-out observations; entering new sensor readings comes in Stage 6.

## Run

With `.venv` activated, install the new scikit-learn dependency:

```powershell
python -m pip install -r requirements-dev.txt
```

Restart the servers if needed. In separate terminals from the repository root:

```powershell
python -m uvicorn backend.main:app --reload
```

```powershell
python -m streamlit run frontend/app.py
```

## What the model learns

Linear Regression estimates an intercept and five coefficients to minimize
the sum of squared training errors. Its prediction is a weighted sum of the
five sensor measurements. It does not receive the hidden synthetic recipe,
squared features, interactions, or concentration as an input.

The synthetic target includes curvature and interactions, so a linear model
will generally leave structured errors, even with target noise set to zero.
This gives us a useful baseline for nonlinear models in Stage 4.

## Follow the request through the code

1. `frontend/model_trainer.py` collects a test fraction and split seed in a form.
   It sends the exact rows currently stored in the browser session's Streamlit
   state, rather than regenerating data from settings.
2. `frontend/api_client.py` sends `POST /models/train` and checks the response.
3. `backend/schemas.py` validates 100–5000 rows, all sensor ranges, a test fraction
   from 0.1 to 0.4, and an integer seed. Invalid inputs receive HTTP 422.
4. `backend/main.py` routes the validated request to the modeling service.
5. `backend/services/modeling.py` splits row indices, builds a pipeline, fits
   it on training rows, and predicts both training and test rows.
6. The frontend stores evaluation results and plots only the test predictions.
   Successful generation clears old results. Failed training keeps the previous
   evaluation with a warning. Captions record the settings actually evaluated.

## Why split before fitting?

The pipeline has two steps:

```text
Training sensors → fit StandardScaler → fit LinearRegression
Test sensors     → use fitted scaler → use fitted regression → test predictions
```

The scaler subtracts each feature's training mean and divides by its training
standard deviation. Scaling is not required for ordinary least squares, but
the pipeline establishes a useful pattern for later scale-sensitive models.
No test rows are used to learn scaling parameters or regression coefficients.
There are no missing values to impute in the synthetic dataset.

The generation seed chooses the observations; the split seed chooses which
observations are held out. Keep both fixed when reproducing an experiment.
The random split suits these independent synthetic snapshots. Real time-series
or batch-correlated data would need a different split strategy.

## Reading the results

| Metric | Meaning |
| --- | --- |
| R² | 1 − sum of squared prediction errors / sum of squared deviations from the test mean. Higher is better; 1 is perfect, 0 matches the test-mean reference, and negative values are possible. |
| RMSE | Square root of mean squared error; more sensitive to large errors. |
| MAE | Mean absolute error; average error magnitude. |

RMSE and MAE use **percentage points**, not relative percentages. An MAE of 2
means an average absolute concentration error of two percentage points. R² is
unitless and is shown as undefined if the evaluated targets are all identical.

Expand the comparison table to see training scores and a separate practical
baseline that always predicts the **training** mean. Its test R² may be negative
because the training mean need not equal the test mean. High training scores
alone do not demonstrate performance on unseen data.

- **Actual vs predicted:** points on the dashed diagonal are perfect. The
  axes use the same scale to make deviations easier to judge.
- **Residual plot:** residual = actual − predicted. Positive means the model
  underpredicts. Curves may suggest missing nonlinear structure; changing
  spread may suggest nonconstant error variance. These plots are diagnostic
  clues, not proof of a particular physical explanation.
- **Held-out table:** row indices are zero-based positions in the displayed
  dataset. Neither the target nor these indices is used as a model feature.

Predictions are not clipped to 0–100: the plots and metrics show the fitted
model's raw behavior. Training does not save a model file or create a shared
global model; the endpoint returns evaluation artifacts for this session.

## Learning exercise

1. Generate 1,000 rows, generation seed 42, and noise 1. Train with a 20% test
   set and split seed 42. Compare the model's test errors with the baseline.
2. Repeat training: scores and test row assignments should be identical.
3. Regenerate with noise 5, keeping both seeds and row count fixed. Retrain and
   compare the errors. Do not assume every individual metric must worsen for
   every possible random sample.
4. Try noise 0. Why does the linear model still make errors?
5. Change a chart selector. Why does this not fit the model again?

This stage uses one holdout split. Repeatedly selecting seeds or models based
on test scores would turn that test set into tuning data. Stage 4 will use
cross-validation on training data for model comparison, while keeping test
data separate. Stage 5 will add a separate interval-calibration split.

Run the checks with `python -m pytest -q`. Tests independently recompute scores,
verify that changing only test labels cannot affect fitted predictions, and
check that the scaler fits only training rows.

References: [scikit-learn LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
and [preventing data leakage](https://scikit-learn.org/stable/common_pitfalls.html).
