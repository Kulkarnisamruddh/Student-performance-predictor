
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "student_performance_model.joblib"
)

model = joblib.load(MODEL_PATH)

# Example student information.
# Replace these values with a student's actual feature values.
student = {
    "school": "GP",
    "sex": "F",
    "age": 17,
    "address": "U",
    "famsize": "GT3",
    "Pstatus": "T",
    "Medu": 4,
    "Fedu": 3,
    "Mjob": "teacher",
    "Fjob": "services",
    "reason": "course",
    "guardian": "mother",
    "traveltime": 1,
    "studytime": 2,
    "failures": 0,
    "schoolsup": "no",
    "famsup": "yes",
    "paid": "no",
    "activities": "yes",
    "nursery": "yes",
    "higher": "yes",
    "internet": "yes",
    "romantic": "no",
    "famrel": 4,
    "freetime": 3,
    "goout": 3,
    "Dalc": 1,
    "Walc": 2,
    "health": 5,
    "absences": 4,
    "G1": 12,
    "G2": 13,
}

student_df = pd.DataFrame([student])

prediction = model.predict(student_df)[0]
prediction = max(0, min(20, prediction))

print(f"Predicted final grade (G3): {prediction:.2f} / 20")
