"""Load and inspect the UCI Student Performance mathematics dataset.

Expected local path: data/student-mat.csv
The raw CSV uses semicolons as separators.
"""
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "student-mat.csv"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}\n"
            "Download the UCI Student Performance dataset, extract it, "
            "and place student-mat.csv inside the project's data/ folder. "
            "See data/README.md for instructions."
        )

    # UCI's CSV is semicolon-separated, so use sep=";".
    df = pd.read_csv(DATA_PATH, sep=";")

    print("\n=== Dataset shape (rows, columns) ===")
    print(df.shape)

    print("\n=== First five rows ===")
    print(df.head())

    print("\n=== Column names ===")
    print(df.columns.tolist())

    print("\n=== Data types ===")
    print(df.dtypes)

    print("\n=== Missing values per column ===")
    print(df.isna().sum())

    print("\n=== Numeric summary ===")
    print(df.describe())

    if "G3" in df.columns:
        print("\n=== Final grade (G3) summary ===")
        print(df["G3"].describe())
        print("\n=== Final grade (G3) value counts ===")
        print(df["G3"].value_counts().sort_index())


if __name__ == "__main__":
    main()
