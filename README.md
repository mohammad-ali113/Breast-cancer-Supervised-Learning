# README — `final_supervised_project.ipynb`

This document explains **what the notebook’s code does**, **how it’s organized**, **how to run it**, and **what results it produces** (findings). It does **not** describe course logistics or the project brief. The details below are derived from the notebook/PDF you shared. 

---

## What the code does (high level)

1. **Loads data** from `sklearn.datasets.load_breast_cancer()` into `pandas` (`X`, `y`).
2. **Quick EDA/quality checks**: class balance bar chart, selected feature box-plots, simple outlier counts (z-score > 3), missing-value scan, summary stats table.
3. **Train/test split** with stratification.
4. **Baseline model**: `DummyClassifier` (most-frequent) for a reference accuracy.
5. **Three candidate models** with consistent preprocessing:

   * `LogisticRegression` in a `Pipeline(StandardScaler → LogisticRegression)`
   * `RandomForestClassifier`
   * `SVC` (RBF kernel) in a `Pipeline(StandardScaler → SVC(probability=True))`
6. **Model selection via 5-fold Stratified CV** (`accuracy`, `f1`, `roc_auc`).
7. **Hyperparameter tuning** with `GridSearchCV` for RF and SVC (score: `roc_auc`).
8. **Final model fit on train**, **evaluation on test**:

   * Accuracy, F1, ROC-AUC
   * Confusion matrix
   * ROC and Precision–Recall curves
9. **Interpretability (lightweight)**:

   * If RF selected: feature importances
   * Else: fit a surrogate `LogisticRegression` and show top |coefficients|

---

## Dependencies

* Python 3.10+ (recommended)
* Core: `numpy`, `pandas`, `matplotlib`
* Modeling: `scikit-learn`, `scipy` (for `zscore`)

Install (CPU-only):

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

---

## How to run

1. Open the notebook in Jupyter or VS Code.
2. **Run all cells in order** (they are stateful).
3. Plots (bar/box/ROC/PR) and tables (stats, CV scores, confusion matrix) render inline.
4. To regenerate identical splits/results, leave `RANDOM_STATE = 42` unchanged.

---

## Code layout (by cell groups)

* **Imports & constants**: libraries, `RANDOM_STATE`, plotting defaults.
* **Load data**: `load_breast_cancer()` → `DataFrame`/`Series`; print shape and class balance.
* **EDA**:

  * Target distribution bar chart.
  * Box-plots for `['mean radius','mean perimeter','mean area','worst area']`.
  * Outlier counts with `scipy.stats.zscore`.
  * Missing-value check (`X.isna().sum()`), `X.describe()` preview.
  * Quick correlation heatmap for top-variance columns.
* **Split**: `train_test_split(..., stratify=y, test_size=0.25)`.
* **Baseline**: `DummyClassifier(strategy="most_frequent")` + `classification_report`.
* **CV comparison**:

  * `LogReg` (scaled), `RF`, `SVC` (scaled)
  * `cross_validate(..., scoring=['accuracy','f1','roc_auc'])`
  * Aggregate to a small `DataFrame` sorted by ROC-AUC.
* **Tuning**:

  * RF grid: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
  * SVC grid (inside pipeline): `C`, `gamma`
  * Pick best by CV ROC-AUC.
* **Final evaluation**:

  * Fit best estimator on train.
  * Test metrics: Accuracy, F1, ROC-AUC.
  * Confusion matrix.
  * `RocCurveDisplay` and `PrecisionRecallDisplay`.
* **Interpretation**:

  * If final is RF: bar chart of top feature importances.
  * Else: fit a surrogate `LogisticRegression` → top absolute coefficients.

---

## Key parameters & choices

* **Stratified CV (k=5)** to respect class balance.
* **Scaling only inside pipelines** for models that need it (LogReg/SVC).
* **Primary model-selection metric**: `roc_auc` during CV/grid search.
* **Probability outputs**: SVC uses `probability=True` to enable ROC/PR plots.

---

## Main findings (from the notebook outputs)

* **Baseline** (majority class): ~**0.629 accuracy**; poor on malignant class—useful only as a floor.
* **Cross-validation** showed very strong performance; **SVC (RBF)** and **LogReg** near the top by ROC-AUC.
* **Best tuned model selected:** **SVC (RBF)** (based on CV AUC).
* **Test set performance** (held-out):

  * **Accuracy ≈ 0.986**
  * **F1 ≈ 0.989**
  * **ROC-AUC ≈ 0.999**
  * **Confusion matrix**: only ~**2 misclassifications** out of 143
* **Feature signals** (surrogate LogReg): high weights among “worst/area/texture/concave-points” families; correlation families (radius/perimeter/area) are evident—regularization helps linear models; trees remain robust without scaling. 

---

## Reproducibility notes

* Randomness controlled via `random_state=42` in split, CV, and models where applicable.
* All charts are generated with `matplotlib` defaults; re-running will reproduce them (subject to minor rendering differences).

---

## Troubleshooting

* **Missing `scipy`**: install `scipy` to enable z-score outlier counts.
* **Matplotlib backend**: if plots don’t show, switch to an inline backend (e.g., `%matplotlib inline` in Jupyter).
* **Performance differences**: make sure scaling remains inside the pipeline; don’t fit scalers on the full dataset.

---

**File referenced:** outputs/figures/metrics and tables captured in the provided PDF snapshot of the notebook. 
