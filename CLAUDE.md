# CLAUDE.md — Resampled Validation and Feature Selection

## Project Overview

This repository provides Python modules for **cross-validation and feature selection with integrated data resampling** for imbalanced classification problems. The core innovation is properly embedding resampling *within* each cross-validation fold, rather than resampling before splitting — which would cause data leakage.

**Problem solved:** scikit-learn's `cross_validate`, `RFE`, and `RFECV` cannot integrate resampling into the fold loop. This library fixes that by re-implementing those workflows with `imbalanced-learn` samplers applied per-fold.

**Primary use cases:** Anomaly detection (manufacturing defect detection, credit card fraud), any classification task with class imbalance.

---

## Repository Structure

```
Resampled-validation-and-feature-selection-/
├── Resampled_learn.py                  # Binary classification module (732 lines)
├── Resampled_learn_multiclass.py       # Multiclass classification module (688 lines)
├── Example_titanic_data.ipynb          # Usage demonstration notebook
├── titanic_train_resample.csv          # Example training data (Titanic dataset)
├── titanic_test_resample.csv           # Example test data
├── README.md                           # Japanese-language README (HTML)
└── 論文/                               # Reference papers directory
    ├── Gene Selection for Cancer Classification using Support Vector Machines.pdf
    ├── Multi-class feature selection for texture classification.pdf
    └── jsm_fsel.pdf
```

No `requirements.txt`, `setup.py`, or CI/CD configuration exists. Dependencies are implicit in the import statements.

---

## Dependencies

Install the following packages (versions are approximate based on the 2018 codebase):

```bash
pip install numpy pandas matplotlib scikit-learn xgboost imbalanced-learn tqdm
```

Key imports used:
- `numpy`, `pandas`, `matplotlib`
- `sklearn.preprocessing.StandardScaler`
- `sklearn.model_selection.StratifiedKFold`
- `sklearn.metrics` — accuracy, precision, recall, f1, roc_auc, log_loss, confusion_matrix
- `sklearn.pipeline.make_pipeline`
- `xgboost.XGBClassifier`
- `imblearn.over_sampling` — `RandomOverSampler`, `SMOTE`
- `imblearn.under_sampling` — `RandomUnderSampler`
- `imblearn.combine` — `SMOTEENN`, `SMOTETomek`
- `imblearn.ensemble` — `BalancedBaggingClassifier`
- `tqdm` — progress bars

**Note:** This codebase uses the older `imbalanced-learn` API (`fit_sample()` instead of `fit_resample()`, `ratio=` instead of `sampling_strategy=`). Modern `imbalanced-learn` (>= 0.5) renamed these. Running with a newer version will produce deprecation warnings or errors.

---

## Module Architecture

### `Resampled_learn.py` — Binary Classification

| Component | Type | Purpose |
|---|---|---|
| `Resampled_Prediction` | class | Wraps resample → scale → fit → predict for a single train/test split |
| `Resampled_Cross_Validate` | class | Stratified k-fold CV with per-fold resampling and scaling |
| `Resampled_Valudation_Score` | function | CV metrics + feature importance extraction (called internally by RFE classes) |
| `Resampled_RFECV` | class | Recursive feature elimination — eliminates down to `n_steps` features, tracking all metrics at each step |
| `Resampled_RFE` | class | Recursive feature elimination — targets a specific final feature count `n_feature_select` |
| `BalancedBagging_Valudation` | class | CV using `BalancedBaggingClassifier` (XGBoost base estimator, hardcoded) |
| `Check_TestData` | function | Final evaluation on held-out test set using `BalancedBaggingClassifier` |
| `get_importance_score` | function | Internal helper — creates a DataFrame mapping feature names to importance scores |

### `Resampled_learn_multiclass.py` — Multiclass Classification

Mirrors `Resampled_learn.py` but with these differences:
- `Resampled_Cross_Validate.__init__` takes `average` parameter (`'micro'`, `'macro'`, `'weighted'`) for sklearn metrics — no `roc_auc_` attribute
- `Resampled_Valudation_Score` takes an `average` parameter; return order is `ACC, F1, PRE, REC, logloss, IM_score` (no `ROC_AUC`)
- `Resampled_RFECV` and `Resampled_RFE` take `average` parameter
- `BalancedBagging_Valudation` takes `average` parameter; `predict()` selects best fold by minimum logloss (binary version uses max accuracy)
- Feature importance for linear models uses `sqrt(sum(coef_^2, axis=0))` to aggregate across classes (binary uses `abs(coef_).ravel()`)
- `StratifiedKFold` uses `random_state=1` (binary version does not set `random_state`)

---

## Core Algorithm: Resampling-Aware Cross-Validation

The critical workflow implemented in every CV class and function:

```
for each fold in StratifiedKFold:
    1. Split → x_train_fold, x_test_fold, y_train_fold, y_test_fold
    2. sampler.fit_sample(x_train_fold, y_train_fold)  # resample train only
    3. scaler.fit(x_resampled)                          # fit scaler on resampled train
    4. x_resampled = scaler.transform(x_resampled)
    5. x_test_fold = scaler.transform(x_test_fold)     # apply to test without refitting
    6. estimator.fit(x_resampled, y_resampled)
    7. y_pred = estimator.predict(x_test_fold)
    8. collect metrics
```

**Important:** `Resampled_Cross_Validate` uses `make_pipeline(scaler, estimator)` instead of manually scaling — the scaler inside the pipeline gets refit on `x_resampled` when `pipe.fit(x_resampled, y_resampled)` is called.

---

## Feature Importance Extraction

`Resampled_Valudation_Score` handles three estimator types:

| Estimator attribute | Example estimators | Extraction method |
|---|---|---|
| `feature_importances_` | XGBoost, Random Forest, LightGBM | Direct use |
| `coef_` (binary) | Logistic Regression, Linear SVM | `abs(coef_).ravel()` |
| `coef_` (multiclass) | Logistic Regression | `sqrt(sum(coef_^2, axis=0))` |
| `dual_coef_` + `support_vectors_` | Kernel SVM | `matmul(dual_coef_, support_vectors_)` then norm |

Importance scores are averaged across all CV folds and returned as a pandas DataFrame with columns `['Var', 'Score']`.

---

## Recursive Feature Elimination (RFE) Classes

### `Resampled_RFECV`

- Eliminates features in steps of `n_steps` from the full feature set down to `n_steps` remaining
- **Constraint:** `len(X.columns)` must be divisible by `n_steps`
- After fitting, accesses results via `mean_score_` and `std_score_` dicts with keys: `'ACC'`, `'ROC_AUC'` (binary only), `'F1'`, `'PRE'`, `'REC'`, `'logloss'`
- `select_num_Q(threshold, score='ROC_AUC')` — returns number of features needed to exceed a threshold metric
- `draw_figure(X, y, ymin, ymax, fill_btw)` — plots score vs. feature count with std shading
- `draw_barchart(X, y)` — horizontal bar chart of feature importances

### `Resampled_RFE`

- Eliminates features down to a target `n_feature_select`
- **Constraint:** `(len(X.columns) - n_feature_select)` must be divisible by `n_steps`
- `support()` method returns a DataFrame with `['Var', 'Support']` boolean mask of selected features

---

## Class Attributes After Fitting

### `Resampled_Cross_Validate`

```python
obj.Matrix       # list of confusion matrices per fold
obj.acc_         # np.array of accuracy per fold
obj.pre_         # np.array of precision per fold
obj.rec_         # np.array of recall per fold
obj.f1_          # np.array of F1 per fold
obj.roc_auc_     # np.array of ROC-AUC per fold (binary only)
obj.logloss_     # np.array of log loss per fold
```

### `Resampled_RFECV` / `Resampled_RFE`

```python
obj.mean_score_  # dict: metric name -> np.array of mean scores across steps
obj.std_score_   # dict: metric name -> np.array of std scores across steps
obj.questions_   # list of DataFrames: feature importance at each step
```

---

## Default Parameters

| Parameter | Binary default | Multiclass default |
|---|---|---|
| `sampler` (Cross_Validate) | `RandomOverSampler(ratio='not minority')` | `RandomUnderSampler(ratio='not minority')` |
| `sampler` (Prediction) | `RandomUnderSampler(ratio='not minority')` | `RandomUnderSampler(ratio='not minority')` |
| `scaler` | `StandardScaler()` | `StandardScaler()` |
| `estimator` | `xgb.XGBClassifier()` | `xgb.XGBClassifier(n_jobs=-1)` |
| `verbose` | `True` | `True` |
| `average` (multiclass) | N/A | `'micro'` |

---

## Known Issues and Limitations

1. **Deprecated imbalanced-learn API:** `fit_sample()` was renamed `fit_resample()` in `imbalanced-learn >= 0.5`. The `ratio` parameter was renamed `sampling_strategy`. Running with modern versions requires updating these calls.

2. **`is` used for value comparison:** In `Resampled_learn_multiclass.py` line 382 and `BalancedBagging_Valudation.predict()` line 549, Python's identity operator `is` is used instead of `==` for comparisons — this is a bug that may produce incorrect results in modern Python.

3. **No unit tests:** Correctness is validated only through the example notebook.

4. **No package structure:** Modules must be imported from the same directory or added to `sys.path`.

5. **`BalancedBagging_Valudation` hardcodes XGBoost:** Unlike other classes, the base estimator is not configurable.

6. **`Resampled_RFE` in multiclass has a missing `PRE_SCORE_std.append`:** Line 658 in `Resampled_learn_multiclass.py` appends to `REC_SCORE_std` but `PRE_SCORE_std` is never populated in the loop.

---

## Usage Pattern

```python
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler

import Resampled_learn as rsl   # for binary classification
# import Resampled_learn_multiclass as rsl  # for multiclass

# Load data (must be pandas DataFrames)
train = pd.read_csv('titanic_train_resample.csv')
X_train = train.drop('Survived', axis=1)
y_train = train['Survived']

# Cross-validation with resampling
cv = rsl.Resampled_Cross_Validate(
    cv=5,
    sampler=RandomOverSampler(ratio='not minority'),
    scaler=StandardScaler(),
    estimator=xgb.XGBClassifier()
)
cv.fit(X_train, y_train)
print(cv.acc_.mean(), cv.roc_auc_.mean())

# Feature selection with RFECV (n_steps must divide len(X_train.columns))
rfecv = rsl.Resampled_RFECV(n_steps=2, cv=5)
rfecv.fit(X_train, y_train)
rfecv.draw_figure(X_train, y_train)

# Feature selection targeting 4 features
rfe = rsl.Resampled_RFE(n_feature_select=4, n_steps=2, cv=5)
rfe.fit(X_train, y_train)
selected = rfe.support()  # DataFrame with 'Var' and 'Support' columns
```

**Important:** `X_train` and `X_test` must be pandas DataFrames (not numpy arrays) because the RFE classes rely on `.columns` and `.loc[:, cols]`.

---

## Development Conventions

- **Encoding:** All files are `# coding: utf-8`; inline comments and string literals are written in Japanese
- **Naming:** Classes use `UpperCamelCase` with underscores (`Resampled_Cross_Validate`); functions use `UpperCamelCase` as well. Note the consistent typo: "Valudation" (not "Validation") in function/class names — preserve this for backward compatibility
- **Warnings:** Both modules suppress all warnings globally via `warnings.filterwarnings('ignore')` at import time
- **Result initialization:** Unfit attributes are initialized to the string `'No Value'` (not `None`)
- **Input format:** All CV classes call `.values` on DataFrame inputs internally — pass pandas DataFrames, not numpy arrays
- **No package `__init__.py`:** Import by placing the `.py` file in the working directory or on `sys.path`

---

## Testing and Validation

There are no automated tests. The primary validation mechanism is `Example_titanic_data.ipynb`, which demonstrates:
1. Loading `titanic_train_resample.csv` (binary imbalanced classification)
2. Running `Resampled_Cross_Validate` to measure baseline CV performance
3. Running `Resampled_RFECV` to find the optimal feature count
4. Reducing features from the full set down to the top 4
5. Final evaluation on `titanic_test_resample.csv`

To validate changes, run the notebook end-to-end and confirm scores are consistent with prior results.

---

## Git Workflow

- Primary development branch: `claude/add-claude-documentation-1HNMh`
- Main branch: `master`
- Push with: `git push -u origin <branch-name>`
