# Student Performance Predictor

A machine-learning project exploring how student and school-related attributes relate to final mathematics grades.

> **Status:** In progress — data exploration and model evaluation are not yet complete.

## Goal

Predict the final mathematics grade (`G3`, on a 0–20 scale) using the UCI Student Performance dataset. Compare a simple baseline with a tree-based model and report results using MAE and RMSE.

## Dataset

- Source: [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)
- File for this project: `student-mat.csv`
- The source dataset contains student grades and demographic, social, and school-related attributes.
- The CSV uses semicolons (`;`) as separators.

Download the dataset from UCI and place `student-mat.csv` in the local `data/` folder. The raw data is intentionally not tracked in Git.

## Important evaluation note

The final grade `G3` is strongly related to earlier-period grades `G1` and `G2`. We will evaluate two settings:
1. **With earlier grades:** a useful comparison, but not an early-warning prediction.
2. **Without `G1` and `G2`:** a harder and more realistic prediction setting when those grades are not yet available.

We will report actual results only after running the experiments.

## Planned workflow

1. Inspect and validate the dataset.
2. Explore grade distributions and relationships.
3. Build a preprocessing pipeline for numeric and categorical features.
4. Compare Linear Regression and Random Forest using a held-out test set.
5. Evaluate MAE and RMSE and discuss limitations.
6. Add a small Streamlit interface after the modelling workflow is reliable.

## Local setup

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Experiments and Results

### Dataset

This project uses the UCI Student Performance dataset for mathematics.
The target variable is `G3`, the final mathematics grade on a 0–20 scale.

The dataset contains 395 student records and 33 columns. The
evaluation uses an 80/20 train/test split with `random_state=42`,
resulting in 316 training records and 79 test records.

### Models

Two regression algorithms were evaluated:

- Linear Regression
- Random Forest Regressor

A mean-prediction baseline was included for comparison.

Two feature sets were tested:

1. **With earlier grades:** all features except `G3`, including `G1` and `G2`.
2. **Without earlier grades:** all features except `G1`, `G2`, and `G3`.

### Results

| Feature set | Model | MAE | RMSE |
|---|---|---:|---:|
| Baseline | Mean predictor | 3.646 | 4.550 |
| With G1 and G2 | Linear Regression | 1.647 | 2.378 |
| With G1 and G2 | Random Forest | 1.198 | 2.035 |
| Without G1 and G2 | Linear Regression | 3.395 | 4.196 |
| Without G1 and G2 | Random Forest | 3.033 | 3.808 |

Lower MAE and RMSE indicate better predictions.


### Cross-Validation

Five-fold cross-validation was performed on the training set
(316 students), using Random Forest and mean absolute error (MAE).

| Feature set | Mean CV MAE | MAE standard deviation |
|---|---:|---:|
| With G1 and G2 | 0.939 | 0.187 |
| Without G1 and G2 | 2.934 | 0.296 |

The model using earlier grades achieved lower average validation
error. The fold-to-fold variation was relatively small for both
feature sets.

Cross-validation scores are distinct from the final held-out test
results. They estimate performance across different validation
subsets of the training data, while the test set is reserved for
final evaluation.


### Observations

- Random Forest achieved lower MAE and RMSE than Linear Regression
  in both feature-set experiments on this test split.
- Earlier grades were highly informative for predicting the final grade.
- Excluding earlier grades made the task more difficult, but Random
  Forest still performed better than the mean baseline.
- The results come from one train/test split and should not be
  interpreted as guaranteed performance on other student populations.

### Limitations

- The dataset is relatively small and may not represent students at
  other schools or in other regions.
- Only one train/test split has been evaluated so far.
- Predictive associations do not establish that a feature causes
  better or worse academic performance.
- Earlier grades may not be available in every intended use case.


## Dataset citation

Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T

This project is educational and should not be used to make consequential decisions about individual students.
