
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Find the project folder and load the dataset.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "student-mat.csv"

df = pd.read_csv(DATA_PATH, sep=";")

sns.set_theme(style="whitegrid")

# 1. Distribution of final grades
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="G3", bins=21, kde=True)
plt.title("Distribution of Final Grades")
plt.xlabel("Final grade (G3)")
plt.ylabel("Number of students")
plt.tight_layout()
plt.show()

# 2. Absences vs final grade
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="absences", y="G3", alpha=0.7)
plt.title("Absences vs Final Grade")
plt.xlabel("Number of absences")
plt.ylabel("Final grade (G3)")
plt.tight_layout()
plt.show()

# 3. Study time vs final grade
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="studytime", y="G3")
plt.title("Study Time vs Final Grade")
plt.xlabel("Weekly study time category")
plt.ylabel("Final grade (G3)")
plt.tight_layout()
plt.show()

# 4. Correlation between earlier and final grades
print("\nGrade correlations:")
print(df[["G1", "G2", "G3"]].corr().round(3))

plt.figure(figsize=(6, 4))
sns.heatmap(
    df[["G1", "G2", "G3"]].corr(),
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
)
plt.title("Correlation Between Grades")
plt.tight_layout()
plt.show()
