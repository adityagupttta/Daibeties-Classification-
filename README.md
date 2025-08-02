# 🧠 Framingham Diabetes Classification using Machine Learning

This project applies machine learning techniques to predict the likelihood of **Type 2 Diabetes** using patient data from the **Framingham Heart Study** dataset. The goal is to build accurate classification models that assist in early diagnosis based on health parameters.

---

## 📌 Problem Statement

Type 2 Diabetes is a major health issue globally. Early prediction based on medical data can help with timely intervention. This project builds and compares multiple ML models for classifying individuals as diabetic or non-diabetic.

---

## 📂 Dataset

- **Source**: Framingham Heart Study
- **Features used**:
  - Age
  - Glucose
  - BMI
  - Blood Pressure (systolic/diastolic)
  - Skin Thickness
  - Insulin
  - Pregnancies
  - Diabetes Pedigree Function
- **Target**: `Outcome` (1 = diabetic, 0 = non-diabetic)

---

## 🧹 Data Preprocessing

- Handled missing values using `.fillna()`
- Feature scaling using `StandardScaler`
- Train-test split: 80% training, 20% testing
- Exploratory visualizations: pairplots, heatmaps, countplots

---

## 🤖 Models Trained

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- XGBoost Classifier

---

## 📈 Evaluation Metrics

- Accuracy Score
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix

| Model                | Accuracy |
|---------------------|----------|
| Logistic Regression | ~78%     |
| Random Forest       | ~81%     |
| XGBoost             | ~84%     |

---

## ⚙️ How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/framingham-diabetes-classification.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter Notebook and open:
   ```bash
   jupyter notebook Glucose_Prediction_Summary.ipynb
   ```

---

## 🔍 Visualizations

- Correlation heatmap
- Class distribution countplots
- Pair plots of input features
- Confusion matrices for each model

---

## 🚀 Future Improvements

- Hyperparameter tuning with GridSearchCV
- Ensemble stacking
- SHAP/LIME for model interpretability
- Deployment with Streamlit or Flask

---

## 📁 Folder Structure (Recommended)

```
framingham-diabetes-classification/
│
├── Glucose_Prediction_Summary.ipynb
├── README.md
├── requirements.txt
├── data/
│   └── diabetes.csv
├── images/
│   └── plots.png
└── models/
    └── saved_model.pkl
```

---

## ✅ Requirements

See `requirements.txt`:

```
matplotlib
numpy
pandas
scikit-learn
seaborn
```
