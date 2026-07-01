import streamlit as st
import pandas as pd

from src.predict import predict_employee

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👨‍💼",
    layout="wide"
)

st.title("👨‍💼 Employee Attrition Prediction")

st.markdown(
    """
Predict whether an employee is likely to leave the company
using a Machine Learning model trained on the IBM HR Analytics dataset.
"""
)

st.divider()

# ==========================================================
# Employee Information
# ==========================================================

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=60,
        value=30
    )

    business_travel = st.selectbox(
        "Business Travel",
        [
            "Travel_Rarely",
            "Travel_Frequently",
            "Non-Travel"
        ]
    )

    daily_rate = st.number_input(
        "Daily Rate",
        min_value=100,
        max_value=1600,
        value=800
    )

    department = st.selectbox(
        "Department",
        [
            "Sales",
            "Research & Development",
            "Human Resources"
        ]
    )

    distance = st.number_input(
        "Distance From Home",
        min_value=1,
        max_value=30,
        value=5
    )

    education = st.selectbox(
        "Education",
        [1, 2, 3, 4, 5]
    )

    education_field = st.selectbox(
        "Education Field",
        [
            "Life Sciences",
            "Medical",
            "Marketing",
            "Technical Degree",
            "Human Resources",
            "Other"
        ]
    )

    environment = st.selectbox(
        "Environment Satisfaction",
        [1, 2, 3, 4]
    )

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )

    hourly_rate = st.number_input(
        "Hourly Rate",
        min_value=30,
        max_value=100,
        value=65
    )

    job_involvement = st.selectbox(
        "Job Involvement",
        [1, 2, 3, 4]
    )

    job_level = st.selectbox(
        "Job Level",
        [1, 2, 3, 4, 5]
    )

with col2:

    job_role = st.selectbox(
        "Job Role",
        [
            "Sales Executive",
            "Research Scientist",
            "Laboratory Technician",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Sales Representative",
            "Research Director",
            "Human Resources"
        ]
    )

    job_satisfaction = st.selectbox(
        "Job Satisfaction",
        [1, 2, 3, 4]
    )

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Divorced"
        ]
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=1000,
        max_value=25000,
        value=6000
    )

    monthly_rate = st.number_input(
        "Monthly Rate",
        min_value=1000,
        max_value=30000,
        value=15000
    )

    num_companies = st.number_input(
        "Number of Companies Worked",
        min_value=0,
        max_value=10,
        value=2
    )

    overtime = st.selectbox(
        "OverTime",
        [
            "No",
            "Yes"
        ]
    )

    salary_hike = st.number_input(
        "Percent Salary Hike",
        min_value=10,
        max_value=30,
        value=15
    )

    performance = st.selectbox(
        "Performance Rating",
        [3, 4]
    )

    relationship = st.selectbox(
        "Relationship Satisfaction",
        [1, 2, 3, 4]
    )

    stock = st.selectbox(
        "Stock Option Level",
        [0, 1, 2, 3]
    )

    total_working_years = st.number_input(
        "Total Working Years",
        min_value=0,
        max_value=40,
        value=10
    )

    training = st.selectbox(
        "Training Times Last Year",
        [0, 1, 2, 3, 4, 5, 6]
    )

    work_life = st.selectbox(
        "Work Life Balance",
        [1, 2, 3, 4]
    )

    years_company = st.number_input(
        "Years At Company",
        min_value=0,
        max_value=40,
        value=5
    )

    years_role = st.number_input(
        "Years In Current Role",
        min_value=0,
        max_value=20,
        value=3
    )

    years_promotion = st.number_input(
        "Years Since Last Promotion",
        min_value=0,
        max_value=15,
        value=1
    )

    years_manager = st.number_input(
        "Years With Current Manager",
        min_value=0,
        max_value=20,
        value=3
    )

st.divider()

predict = st.button(
    "Predict Attrition",
    use_container_width=True
)

# ==========================================================
# Prediction
# ==========================================================

if predict:

    input_df = pd.DataFrame([{

        "Age": age,
        "BusinessTravel": business_travel,
        "DailyRate": daily_rate,
        "Department": department,
        "DistanceFromHome": distance,
        "Education": education,
        "EducationField": education_field,
        "EnvironmentSatisfaction": environment,
        "Gender": gender,
        "HourlyRate": hourly_rate,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobRole": job_role,
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital_status,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_companies,
        "OverTime": overtime,
        "PercentSalaryHike": salary_hike,
        "PerformanceRating": performance,
        "RelationshipSatisfaction": relationship,
        "StockOptionLevel": stock,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training,
        "WorkLifeBalance": work_life,
        "YearsAtCompany": years_company,
        "YearsInCurrentRole": years_role,
        "YearsSinceLastPromotion": years_promotion,
        "YearsWithCurrManager": years_manager

    }])

    prediction, probability = predict_employee(input_df)

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            "⚠️ This employee is likely to leave the company."
        )

    else:

        st.success(
            "✅ This employee is likely to stay in the company."
        )

    if probability is not None:

        st.write(
            f"### Probability of Attrition: {probability:.2%}"
        )

        st.progress(float(probability))

        if probability >= 0.75:

            st.error("High Risk of Attrition")

        elif probability >= 0.50:

            st.warning("Moderate Risk of Attrition")

        else:

            st.success("Low Risk of Attrition")

    st.divider()

    st.subheader("Employee Details Used")

    st.dataframe(
        input_df,
        use_container_width=True,
        hide_index=True
    )

st.markdown("---")

st.caption(
    "Employee Attrition Prediction using Machine Learning | IBM HR Analytics Dataset"
)
