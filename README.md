# Student Performance Predictor 🎓

A machine learning web application that predicts students' final mathematics grades using demographic, social, and academic features. Built with Python, scikit-learn, pandas, and Streamlit.

## 🚀 Live Demo

**Try the app here:** [Student Performance Predictor](https://student-performance-predictor-h6hejiqq5tyzkmnfbe28jf.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://student-performance-predictor-h6hejiqq5tyzkmnfbe28jf.streamlit.app/)

## 🚀 Project Overview

This project explores student performance data and trains regression models to estimate final mathematics grades (`G3`) on a scale of 0–20.

It also provides an interactive dashboard for exploring grade patterns, comparing model performance, and generating predictions from user-provided student information.

## ✨ Features

- **Grade prediction:** Predicts a student's final mathematics grade using a trained Random Forest regression model.
- **Model evaluation:** Displays Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- **Data visualizations:** Explores study time, previous grades, and the distribution of final grades.
- **Interactive data explorer:** Filters student records by gender and study-time category.
- **Web interface:** Provides a user-friendly dashboard built with Streamlit.

## 🛠️ Tech Stack

- Python
- pandas and NumPy
- scikit-learn
- Streamlit
- Joblib
- Matplotlib/Streamlit charting, where applicable

## 📊 Dataset

The project uses the **Student Performance** dataset from the UCI Machine Learning Repository, specifically the mathematics dataset (`student-mat.csv`).

- **Observations:** 395 students
- **Input features:** 32
- **Target variable:** `G3` — final mathematics grade
- **Grade range:** 0–20

The dataset contains student demographic, family, lifestyle, and academic information.

Dataset source: https://archive.ics.uci.edu/dataset/320/student+performance

## 🤖 Model Development

The following approaches were evaluated using the same 80/20 train-test split (`random_state=42`):

- Mean baseline regressor
- Linear Regression
- Random Forest Regressor

Two feature configurations were compared:
1. **With `G1` and `G2`:** Includes first- and second-period grades.
2. **Without `G1` and `G2`:** Excludes earlier grades to assess prediction using other available information.

The selected Random Forest model uses 200 trees, `random_state=42`, and `min_samples_leaf=2`.

## 📈 Test-Set Results

The test set contains 79 students, with 316 students in the training set.

| Feature set | Model | MAE | RMSE |
|---|---|---:|---:|
| Baseline | Mean predictor | 3.646 | 4.550 |
| With G1 and G2 | Linear Regression | 1.647 | 2.378 |
| With G1 and G2 | Random Forest | **1.198** | **2.035** |
| Without G1 and G2 | Linear Regression | 3.395 | 4.196 |
| Without G1 and G2 | Random Forest | 3.033 | 3.808 |

**MAE** measures the average absolute difference between predicted and actual grades. **RMSE** penalizes larger errors more heavily.

### Five-Fold Cross-Validation

Cross-validation was performed on the training set for the Random Forest model.

| Feature set | Mean CV MAE | Standard deviation |
|---|---:|---:|
| With G1 and G2 | 0.939 | 0.187 |
| Without G1 and G2 | 2.934 | 0.296 |

These results show that earlier grades provide substantial predictive information. Performance on future students may differ from the reported results.

## 📁 Project Structure

```text
student-performance-predictor/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── student-mat.csv
├── models/
│   ├── student_performance_model.joblib
│   └── evaluation_metrics.json
└── src/
    ├── explore_data.py
    ├── predict.py
    └── train_model.py
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Kulkarnisamruddh/Student-performance-predictor.git
cd Student-performance-predictor
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The app should open in your browser.

## 🔁 Retraining the Model

To train the model again and regenerate the saved evaluation metrics:

```bash
python src/train_model.py
```

This creates or updates the model and evaluation metrics in the `models/` directory.

## ☁️ Deployment

The application can be deployed using Streamlit Community Cloud:

1. Push the project to GitHub.
2. Open https://share.streamlit.io/.
3. Connect your GitHub account.
4. Select this repository and the `main` branch.
5. Set the main file path to `app.py`.
6. Deploy the app.

Ensure the dataset, trained model, metrics file, and dependencies are available to the deployed app.

**Live demo:** Add your Streamlit app URL here after deployment.

## ⚠️ Limitations

- The dataset is relatively small and may not represent all student populations.
- The predictions are estimates, not guaranteed final grades.
- Including `G1` and `G2` improves performance but means the model relies on earlier academic results.
- The reported test and cross-validation results do not guarantee equivalent performance on new data.
- Observational relationships should not be interpreted as proof of causation.

## 🎯 Future Improvements

- Add actual-versus-predicted grade visualizations.
- Compare additional regression algorithms.
- Add model error analysis and feature importance.
- Improve input validation and automated tests.
- Deploy the app and include the live demo link.

## 👨‍💻 Author

**Samruddh Kulkarni**

GitHub: https://github.com/Kulkarnisamruddh

---

*Built as a machine learning project to practice data exploration, regression modeling, model evaluation, and deployment.*
