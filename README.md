# 👨‍💼 Employee Attrition Prediction System

A complete Machine Learning project that predicts whether an employee is likely to leave a company using IBM HR Analytics dataset.

This project demonstrates an end-to-end ML pipeline:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training & selection
- Evaluation metrics
- Streamlit web app deployment

---

# 📊 Dataset

**Source:** IBM HR Analytics Employee Attrition & Performance (Kaggle)

- 1470 records
- 35 features
- Target: Attrition (Yes / No)

---

# 🚀 Workflow

1. Load dataset
2. Clean & preprocess data
3. Encode categorical variables
4. Scale features
5. Train multiple ML models
6. Select best model
7. Evaluate performance
8. Deploy using Streamlit

---

# 🧠 Models Used

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- KNN
- SVM

---

# 🏆 Best Model

The best performing model is automatically saved as:

```
models/best_model.pkl
```

---

# 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- Confusion Matrix

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
git clone https://github.com/your-username/employee-attrition-prediction.git
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

# 🖥️ Streamlit App Features

- Input employee details
- Predict attrition (Yes/No)
- Shows probability score
- Risk classification:
  - Low Risk
  - Medium Risk
  - High Risk

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

Built as a portfolio project for Machine Learning / Data Science roles.

---

# 📜 License

This project is open-source for educational use.
