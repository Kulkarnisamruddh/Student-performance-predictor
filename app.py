
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st


# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "student_performance_model.joblib"

PROJECT_ROOT = Path(__file__).resolve().parent
METRICS_PATH = PROJECT_ROOT / "models" / "evaluation_metrics.json"

with open(METRICS_PATH, "r", encoding="utf-8") as file:
    metrics = json.load(file)

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📚",
    layout="centered",
)

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Student Performance Predictor")

st.markdown(
    """
    Predict mathematics grades using machine learning
    and explore patterns in student performance.

    **Explore the data. Understand the model. Generate a prediction.**
    """
)
st.write(
    "Estimate a student's final mathematics grade using "
    "a machine-learning model trained on student performance data."
)

st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Mean Absolute Error (MAE)",
        f'{metrics["test_mae"]:.3f}',
    )

with col2:
    st.metric(
        "Root Mean Squared Error (RMSE)",
        f'{metrics["test_rmse"]:.3f}',
    )

st.caption(
    "Metrics are measured on a held-out test set of 79 students. "
    "They describe evaluation performance, not a guarantee for "
    "individual predictions."
)

DATA_PATH = PROJECT_ROOT / "data" / "student-mat.csv"

df = pd.read_csv(DATA_PATH, sep=";")

st.info(
    "This model uses G1 and G2 (earlier grades). "
    "The prediction is an estimate, not a guaranteed result."
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found. Run `python src/train_model.py` "
        "to generate the trained model first."
    )
    st.stop()


with st.form("student_inputs"):
    st.subheader("Student information")

    col1, col2 = st.columns(2)

    with col1:
        school = st.selectbox("School", ["GP", "MS"])
        sex = st.selectbox("Sex", ["F", "M"])
        age = st.number_input("Age", min_value=15, max_value=22, value=17)
        address = st.selectbox("Home address type", ["U", "R"])
        famsize = st.selectbox("Family size", ["GT3", "LE3"])
        Pstatus = st.selectbox("Parents living together", ["T", "A"])
        Medu = st.selectbox("Mother's education (0–4)", [0, 1, 2, 3, 4])
        Fedu = st.selectbox("Father's education (0–4)", [0, 1, 2, 3, 4])
        Mjob = st.selectbox(
            "Mother's job",
            ["teacher", "health", "services", "at_home", "other"],
        )
        Fjob = st.selectbox(
            "Father's job",
            ["teacher", "health", "services", "at_home", "other"],
        )
        reason = st.selectbox(
            "Reason for choosing school",
            ["course", "home", "reputation", "other"],
        )
        guardian = st.selectbox("Guardian", ["mother", "father", "other"])
        traveltime = st.selectbox("Travel time (1–4)", [1, 2, 3, 4])
        studytime = st.selectbox("Study time (1–4)", [1, 2, 3, 4])
        failures = st.selectbox("Past class failures (0–3)", [0, 1, 2, 3])
        absences = st.number_input(
            "School absences",
            min_value=0,
            max_value=100,
            value=4,
        )
        G1 = st.number_input(
            "First-period grade (G1)",
            min_value=0,
            max_value=20,
            value=12,
        )
        G2 = st.number_input(
            "Second-period grade (G2)",
            min_value=0,
            max_value=20,
            value=13,
        )

    with col2:
        schoolsup = st.selectbox("Extra school support", ["yes", "no"])
        famsup = st.selectbox("Family educational support", ["yes", "no"])
        paid = st.selectbox("Extra paid classes", ["yes", "no"])
        activities = st.selectbox("Extracurricular activities", ["yes", "no"])
        nursery = st.selectbox("Attended nursery school", ["yes", "no"])
        higher = st.selectbox("Wants higher education", ["yes", "no"])
        internet = st.selectbox("Internet access at home", ["yes", "no"])
        romantic = st.selectbox("In a romantic relationship", ["yes", "no"])
        famrel = st.selectbox("Family relationship quality (1–5)", [1, 2, 3, 4, 5])
        freetime = st.selectbox("Free time (1–5)", [1, 2, 3, 4, 5])
        goout = st.selectbox("Going out with friends (1–5)", [1, 2, 3, 4, 5])
        Dalc = st.selectbox("Workday alcohol consumption (1–5)", [1, 2, 3, 4, 5])
        Walc = st.selectbox("Weekend alcohol consumption (1–5)", [1, 2, 3, 4, 5])
        health = st.selectbox("Health status (1–5)", [1, 2, 3, 4, 5])

    submitted = st.form_submit_button("Predict final grade", type="primary")


if submitted:
    student = {
        "school": school,
        "sex": sex,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Pstatus": Pstatus,
        "Medu": Medu,
        "Fedu": Fedu,
        "Mjob": Mjob,
        "Fjob": Fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": Dalc,
        "Walc": Walc,
        "health": health,
        "absences": absences,
        "G1": G1,
        "G2": G2,
    }

    student_df = pd.DataFrame([student])
    prediction = float(model.predict(student_df)[0])
    prediction = max(0.0, min(20.0, prediction))

    st.subheader("Predicted final grade")
    st.metric("Estimated G3", f"{prediction:.2f} / 20")
    st.progress(prediction / 20)

    
st.header("Explore Student Data")

st.write(
    "Explore how study time and previous grades relate "
    "to students' final mathematics grades."
)


with st.sidebar:
    st.title("🎓 Student Predictor")

    st.write(
        "An interactive machine learning project "
        "that explores and predicts mathematics grades."
    )

    st.divider()

    st.subheader("Project Information")
    st.write("**Model:** Random Forest Regressor")
    st.write("**Dataset:** UCI Student Performance")
    st.write("**Target:** Final grade (G3)")
    st.write("**Scale:** 0–20")

    st.divider()

    st.caption(
        "Educational project only. Predictions are estimates, "
        "not guarantees of student performance."
    )

# Chart 1: Study time vs final grade
st.subheader("Study Time vs Final Grade")

study_summary = (
    df.groupby("studytime", as_index=False)["G3"]
    .mean()
)

study_labels = {
    1: "<2 hours/week",
    2: "2–5 hours/week",
    3: "5–10 hours/week",
    4: ">10 hours/week",
}

study_summary["Study time"] = (
    study_summary["studytime"].map(study_labels)
)

study_summary = study_summary.rename(
    columns={"G3": "Average final grade"}
)

st.bar_chart(
    study_summary,
    x="Study time",
    y="Average final grade",
)

st.caption(
    "Study time is grouped into categories. "
    "The chart shows average final grades in each category."
)

# Chart 2: Previous grades vs final grade
st.subheader("Previous Grades vs Final Grade")

st.scatter_chart(
    df,
    x="G2",
    y="G3",
)

st.caption(
    "Each point represents one student. G2 is the second-period "
    "grade, and G3 is the final grade."
)


st.caption(
        "This is a statistical estimate based on historical data. "
        "It should not be used as the sole basis for decisions about a student."
    )


# Chart 3: Distribution of final grades
st.subheader("Distribution of Final Grades")

st.write(
    "This chart shows how frequently each final grade "
    "appears in the dataset."
)

grade_counts = (
    df["G3"]
    .value_counts()
    .sort_index()
)

st.bar_chart(
    grade_counts,
    x_label="Final Grade (G3)",
    y_label="Number of Students",
)

st.caption(
    "Grades range from 0 to 20. A taller bar means more "
    "students received that grade."
)


# Interactive Student Data Explorer
st.header("Interactive Data Explorer")

st.write(
    "Filter the dataset to explore different groups of students."
)

# Filters
sex_options = ["All"] + sorted(df["sex"].dropna().unique().tolist())

selected_sex = st.selectbox(
    "Filter by gender",
    sex_options,
    key="explorer_sex",
)

study_options = sorted(df["studytime"].dropna().unique().tolist())

selected_study = st.selectbox(
    "Filter by study time",
    ["All"] + study_options,
    format_func=lambda value: (
        "All study-time groups"
        if value == "All"
        else {
            1: "Less than 2 hours/week",
            2: "2–5 hours/week",
            3: "5–10 hours/week",
            4: "More than 10 hours/week",
        }.get(value, str(value))
    ),
    key="explorer_study",
)

# Apply filters
filtered_df = df.copy()

if selected_sex != "All":
    filtered_df = filtered_df[
        filtered_df["sex"] == selected_sex
    ]

if selected_study != "All":
    filtered_df = filtered_df[
        filtered_df["studytime"] == selected_study
    ]

# Show summary
st.metric("Students matching filters", len(filtered_df))

if not filtered_df.empty:
    st.subheader("Final Grades in Selected Group")

    st.bar_chart(
        filtered_df["G3"].value_counts().sort_index()
    )

    st.subheader("Student Records")

    st.dataframe(
        filtered_df[["sex", "studytime", "G1", "G2", "G3"]],
        use_container_width=True,
    )
else:
    st.info("No students match the selected filters.")
