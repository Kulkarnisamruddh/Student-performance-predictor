from pathlib import Path
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, cross_val_score


# 1. Load the dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "student-mat.csv"

df = pd.read_csv(DATA_PATH, sep=";")

target = "G3"
y = df[target]

# 2. Define the two feature sets
feature_sets = {
    "With G1 and G2": df.drop(columns=["G3"]),
    "Without G1 and G2": df.drop(columns=["G1", "G2", "G3"]),
}

# 3. Make one shared train/test split
train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.2,
    random_state=42,
)

y_train = y.loc[train_idx]
y_test = y.loc[test_idx]

print(f"Training rows: {len(train_idx)}")
print(f"Testing rows: {len(test_idx)}")

results = []

# 4. Evaluate a simple mean baseline
baseline_predictions = [y_train.mean()] * len(y_test)

results.append({
    "Feature set": "Baseline",
    "Model": "Mean predictor",
    "MAE": mean_absolute_error(y_test, baseline_predictions),
    "RMSE": mean_squared_error(
        y_test, baseline_predictions
    ) ** 0.5,
})

# 5. Evaluate each model on each feature set
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=2,
    ),
}

for feature_set_name, X in feature_sets.items():
    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]

    numeric_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                ]),
                categorical_columns,
            ),
        ]
    )

    for model_name, regressor in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        results.append({
            "Feature set": feature_set_name,
            "Model": model_name,
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": mean_squared_error(
                y_test, predictions
            ) ** 0.5,
        })

# 6. Display results in a comparison table
results_df = pd.DataFrame(results)

print("\nModel Comparison")
print(
    results_df.to_string(
        index=False,
        formatters={
            "MAE": "{:.3f}".format,
            "RMSE": "{:.3f}".format,
        },
    )
)


# Cross-validation on training data only
cv = KFold(n_splits=5, shuffle=True, random_state=42)

print("\nRandom Forest: 5-Fold Cross-Validation")

for feature_set_name, X in feature_sets.items():
    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]

    numeric_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                ]),
                categorical_columns,
            ),
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                min_samples_leaf=2,
            ),
        ),
    ])

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_absolute_error",
    )

    mae_scores = -scores

    print(f"\n{feature_set_name}")
    print(f"Fold MAEs: {np.round(mae_scores, 3)}")
    print(f"Mean MAE:  {mae_scores.mean():.3f}")
    print(f"Std MAE:   {mae_scores.std():.3f}")


# Train the final model and save it
# This version uses G1 and G2 (earlier grades).

final_features = df.drop(columns=["G3"])
X_train = final_features.loc[train_idx]
y_train = y.loc[train_idx]

numeric_columns = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            SimpleImputer(strategy="median"),
            numeric_columns,
        ),
        (
            "categorical",
            Pipeline(steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]),
            categorical_columns,
        ),
    ]
)

final_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            min_samples_leaf=2,
        ),
    ),
])

final_model.fit(X_train, y_train)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


MODEL_PATH = MODEL_DIR / "student_performance_model.joblib"
joblib.dump(final_model, MODEL_PATH)

print(f"\nSaved trained model to: {MODEL_PATH}")

# Save evaluation metrics for the Streamlit app
rf_result = results_df[
    (results_df["Feature set"] == "With G1 and G2")
    & (results_df["Model"] == "Random Forest")
].iloc[0]

metrics = {
    "model": "Random Forest",
    "test_mae": float(rf_result["MAE"]),
    "test_rmse": float(rf_result["RMSE"]),
}

METRICS_PATH = MODEL_DIR / "evaluation_metrics.json"

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print(f"Saved evaluation metrics to: {METRICS_PATH}")

