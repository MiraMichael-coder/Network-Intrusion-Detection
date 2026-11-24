# Logistic Regression Pipeline — Explained

This document walks through the logistic regression (LR) baseline for the
UNSW-NB15 network intrusion detection project: how the data flows through the
pipeline, how it is evaluated, tuned, and interpreted.

## Overview

- **Task:** binary classification — predict `label` (`1` = attack, `0` = normal)
  from UNSW-NB15 traffic records.
- **Data:** 175,341 training rows (`UNSW_NB15_training-set.csv`) and 82,332
  official test rows (`UNSW_NB15_testing-set.csv`).
- **Why LR first:** it is fast, fully interpretable (coefficients map directly
  to features), and a strong sanity baseline. Any later model (trees, ensembles,
  neural nets) must beat it to be worth the extra complexity.
- **Model:** `LogisticRegression` with the `saga` solver, `tol=1e-3`,
  `max_iter=1000`, `random_state=SEED` (`src/models.py`).

## The Pipeline

`make_pipeline("lr", X)` in `src/models.py` returns an sklearn `Pipeline` with
two steps:

```
Pipeline([
    ("prep",  ColumnTransformer(...)),   # preprocessing
    ("model", LogisticRegression(...)),   # classifier
])
```

### Preprocessing — `create_preprocessor` (`src/preprocessing.py`)

A `ColumnTransformer` routes each column to the right transform:

- **Numerical columns** (everything except `CAT_COLS`) → `StandardScaler`
  (zero mean, unit variance).
- **Categorical columns** `CAT_COLS = ["proto", "service", "state"]` →
  `OneHotEncoder(handle_unknown="ignore")`. `handle_unknown="ignore"` means a
  category seen at test time but not in training produces all-zero columns
  instead of crashing.

### Excluded columns — `EXCLUDED_COLS`

```python
EXCLUDED_COLS = ["id", "attack_cat", "label"]
```

- `id` — row identifier, no predictive value.
- `attack_cat` — the attack *type* (e.g. "DoS", "Fuzzers"); it is a label
  annotation, not a feature, and would leak the answer.
- `label` — this is the target `y`, dropped from `X`.

### Why preprocessing lives inside the Pipeline

The preprocessor is a pipeline step, not a separate fit-then-transform step.
This means every time the pipeline is fit — on the full training set, on each
CV fold, or on a `GridSearchCV` candidate — the scaler and encoder are
**re-fit only on that fold's training portion**. Test-fold statistics never
leak into training, which is what makes CV scores honest.

## Evaluation — `src/evaluate.py`

### `cv_evaluate(name, pipeline, X, y)` — 5-fold stratified CV

This is the main evaluation entry point. It:

1. Defines a `scoring` dict: `accuracy`, `precision`, `recall`, `f1`,
   `roc_auc`, `average_precision` (built-in sklearn scorers) plus `fpr`, which
   has no built-in scorer so it is wrapped with `make_scorer(_fpr)`.
2. Runs `cross_validate` with `StratifiedKFold(5, shuffle=True,
   random_state=SEED)` and `n_jobs=-1` (folds run in parallel).
3. For each fold, sklearn fits a **fresh copy** of the pipeline on 4/5 of the
   data and scores it on the held-out 1/5. Stratification keeps the
   attack/normal ratio identical across folds.
4. Averages the 5 fold scores into `mean` and `std` columns and appends one row
   with `split="cv"` to `results/metrics.csv`.

The `*_std` columns tell you how stable the metric is across folds — small std
means the score is reliable, large std means it depends heavily on which rows
land in the test fold.

### `evaluate(name, fitted_model, X, y, split)` — single-split scoring

Used when you already have a fitted model and want to score it on one dataset
(e.g. a held-out split or the official test set):

1. Times `predict(X)` → `predict_seconds`.
2. Gets continuous scores for ROC/PR AUC: `predict_proba(X)[:, 1]` if
   available, otherwise `decision_function(X)`.
3. Computes the confusion matrix (`labels=[0, 1]` → rows are true
   normal/attack, columns predicted normal/attack), derives `fpr =
   fp / (fp + tn)`, and computes accuracy, precision, recall, F1, ROC-AUC,
   PR-AUC.
4. Saves a confusion-matrix PNG to `results/figures/cm_<name>_<split>.png`
   (matplotlib runs with `Agg` backend — files only, no window).
5. Appends one row to `results/metrics.csv`.

## Hyperparameter Tuning

Done in `notebooks/02_lr.ipynb` with `GridSearchCV`:

```python
grid = {"model__C": [0.01, 0.1, 1, 10, 100],
        "model__penalty": ["l1", "l2"],
        "model__class_weight": [None, "balanced"]}
search = GridSearchCV(pipe, grid, scoring="f1", n_jobs=-1,
                      cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
```

- **Grid:** 5 `C` values × 2 penalties × 2 class weights = 20 candidates.
- **Scoring:** `f1` — chosen because the classes are imbalanced and F1 balances
  precision and recall.
- **CV:** the same 5-fold stratified split as `cv_evaluate`, so tuning and
  evaluation see identical folds.
- **Best params** are written to `configs/lr.json`:

```json
{"model__C": 10, "model__class_weight": null, "model__penalty": "l2"}
```

The `model__` prefix is required because the parameters belong to the
`"model"` step of the pipeline (step names are `"prep"` and `"model"` so grid
keys look like `model__C`).

## Coefficient Interpretation

After tuning, the notebook extracts the fitted pipeline's coefficients:

```python
best.named_steps["prep"].get_feature_names_out()   # expanded feature names
best.named_steps["model"].coef_[0]                  # one weight per feature
```

`get_feature_names_out()` returns names like `num__dttl`, `cat__state_INT`,
`cat__proto_udp` — the `num__`/`cat__` prefix tells you which transformer
produced the column, and the suffix is the original column or one-hot category.
The top 15 by absolute coefficient are saved to
`results/lr_coefficients.csv`. Positive coefficients push toward "attack",
negative toward "normal". Example from the actual run: `num__dttl` (+4.50) and
`cat__state_INT` (+3.79) are the strongest attack indicators; `num__swin`
(−3.57) and `cat__proto_udp` (−3.00) push toward normal.

## Out-of-Fold (OOF) Probabilities — `src/oof.py`

`save_oof("lr")` produces `results/oof_lr.csv` with columns
`id, label, proba` — one row per training sample.

- It loads the best params from `configs/lr.json`, rebuilds the pipeline via
  `make_pipeline`, and runs `cross_val_predict(..., method="predict_proba")`
  with the **same** `StratifiedKFold(5, shuffle=True, random_state=SEED)` used
  everywhere else.
- Each row's `proba` is the model's predicted P(attack) from the fold where
  that row was **held out** — the model never saw that row during training.

**Why OOF matters:**

- These are *honest* per-row predictions. If you trained one model on all the
  data and predicted on the same data, the probabilities would be overconfident
  (the model memorized them).
- They line up exactly with the reported CV metrics, since the folds are
  identical.
- They enable downstream work: threshold analysis, calibration plots, error
  analysis per attack type, and stacking/ensembling later models on top of LR's
  OOF scores.

## How to Read the Metrics

From the actual run (`results/metrics.csv`), tuned LR (`lr_tuned`, split=`cv`):

| Metric | Value | Meaning |
|---|---|---|
| accuracy | 0.9332 | 93.3% of rows classified correctly — but misleading alone on imbalanced data |
| precision | 0.9186 | Of rows flagged as attack, 91.9% really are attacks |
| recall | 0.9895 | Of real attacks, 98.9% are caught — very few attacks missed |
| f1 | 0.9527 | Harmonic mean of precision and recall |
| fpr | 0.1868 | 18.7% of normal traffic is falsely flagged as attack |
| roc_auc | 0.9821 | Probability a random attack scores higher than a random normal |
| pr_auc | 0.9906 | Area under precision-recall curve; robust to class imbalance |

Key readings:

- **Recall ≫ precision trade-off:** LR catches almost all attacks (0.99 recall)
  at the cost of flagging ~19% of normal traffic (FPR 0.19). For intrusion
  detection this bias toward catching attacks is usually desirable, but the
  false-alarm rate may need a higher decision threshold in deployment.
- **`split="cv"` vs train/test rows:** `cv` rows are 5-fold cross-validation
  estimates (with `*_std` columns); single-split rows have `split` set to e.g.
  `"train"` or `"test"` and no std columns. CV is the trustworthy number;
  a single train score is optimistic.
- **Dummy baseline:** `dummy_most_frequent` shows what "always predict the
  majority class" achieves — recall 1.0 but FPR 1.0 and ROC-AUC 0.5. LR beats
  it on every meaningful metric.
- **Tuning gain is small:** `lr_tuned` (F1 0.95275) barely beats
  `lr_baseline` (F1 0.95264) — LR was already near its ceiling on this data.

## Project Rules

- **No test-set leakage:** the official test CSV is never used for training,
  tuning, or preprocessing decisions. All fitting happens inside CV folds or
  on the training set only.
- **Preprocessing inside the pipeline:** scalers/encoders are always pipeline
  steps so they re-fit per fold — never fit on the full data beforehand.
- **Shared seed:** `SEED = 42` in `src/config.py` is used for CV splits, model
  `random_state`, and anything stochastic, so results are repeatable.
- **One config, one place:** paths, seeds, and categorical columns live in
  `src/config.py`; every other file imports from there.
