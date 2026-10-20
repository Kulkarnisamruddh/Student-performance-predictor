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

## Dataset citation

Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T

This project is educational and should not be used to make consequential decisions about individual students.
