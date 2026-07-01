# 👨‍💼 Employee Attrition Analysis & Prediction System

An end-to-end **Machine Learning and Data Science project** that analyzes employee attrition patterns and builds predictive models using the IBM HR Analytics dataset.

The project emphasizes **hypothesis-driven exploratory data analysis (EDA)**, statistical validation, and a production-style ML pipeline with deployment.

---

# 📊 Dataset

- Source: IBM HR Analytics Employee Attrition Dataset
- Records: 1470 employees
- Features: 35 attributes
- Target Variable: Attrition (Yes/No)

---

# 🎯 Objectives

- Understand key drivers of employee attrition  
- Validate hypotheses using statistical analysis  
- Build predictive machine learning models  
- Deploy a real-time inference system  

---

# 🧪 Exploratory Data Analysis (EDA)

The analysis includes both visual and statistical exploration of workforce behavior.

## Key Analyses:
- Attrition distribution analysis  
- OverTime vs Attrition relationship  
- Income and tenure-based patterns  
- Correlation heatmap of numerical features  

## Hypotheses Tested:
- H1: Employees working overtime have higher attrition rates  
- H2: Lower income is associated with higher attrition  
- H3: Employees with fewer years at the company are more likely to leave  

## Statistical Validation:
- Chi-square test for independence between categorical variables  
- p-value analysis to determine statistical significance of OverTime vs Attrition relationship  

---

# 📈 Feature Insights

Feature importance analysis identified key drivers of attrition:

- OverTime  
- MonthlyIncome  
- JobSatisfaction  
- YearsAtCompany  
- WorkLifeBalance  

---

# 🤖 Machine Learning Models

The following classification models were evaluated:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- K-Nearest Neighbors  
- Support Vector Machine  

---

# 🏆 Model Selection Strategy

Models were compared using:

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  

The best-performing model was selected and serialized for deployment.

---

# 🏗️ System Architecture

Raw Data  
→ Data Preprocessing (Encoding + Scaling)  
→ Exploratory Data Analysis + Hypothesis Testing  
→ Model Training & Evaluation  
→ Best Model Selection  
→ Model Serialization (Scaler + Feature Pipeline + Model)  
→ Streamlit Deployment  

---

# 📊 Evaluation Metrics

- Confusion Matrix  
- Classification Report  
- ROC Curve  
- Feature Importance Analysis  

---

# 🧠 Key Technical Contributions

- Built a reproducible end-to-end ML pipeline with modular architecture  
- Ensured consistent preprocessing between training and inference  
- Introduced hypothesis-driven EDA with statistical validation  
- Implemented automated model comparison and selection pipeline  
- Developed real-time Streamlit-based prediction system  

---

# 🖥️ Streamlit Application

Features:

- Input employee attributes dynamically  
- Predict attrition probability  
- Risk classification (Low / Medium / High)  
- Real-time probability visualization  

---

# 📁 Project Structure

```
employee-attrition-prediction/
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── images/
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── roc_curve.png
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── notebooks/
│   └── Employee_Attrition_Analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

```bash
git clone https://github.com/KhushpreetKaur10/employee-attrition-prediction.git
cd employee-attrition-prediction
```

```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux
```

```bash
pip install -r requirements.txt
```

---

# 🏃 Run Project

```bash
python -m src.train
```

```bash
python -m src.evaluate
```

```bash
streamlit run app.py
```

---


# 📸 Outputs

- Class distribution plot
- Correlation heatmap
- Confusion matrix
- ROC curve
- Feature importance chart

---

# 🔥 Key Highlights

- End-to-end ML pipeline
- Clean modular architecture
- Multiple ML models comparison
- Automatic best model selection
- Real-time prediction web app
- Industry-style project structure

---

# 🚀 Future Improvements

- Hyperparameter tuning (GridSearchCV)
- SHAP explainability
- Docker deployment
- Cloud hosting (AWS / Azure / Streamlit Cloud)
- API integration using FastAPI

---

# 👨‍💻 Author

Khushpreet Kaur

---

# 📜 License

This project is open-source for educational use.
