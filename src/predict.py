import joblib
import pandas as pd

from src.config import BEST_MODEL_PATH
from src.preprocess import (
    get_processed_data,
    preprocess_input
)


# =====================================================
# Load Model
# =====================================================

def load_model():
    """
    Load the trained machine learning model.
    """
    model = joblib.load(BEST_MODEL_PATH)
    return model


# =====================================================
# Predict for New Employee
# =====================================================

def predict_employee(input_df):
    """
    Predict employee attrition for new user input.

    Parameters
    ----------
    input_df : pandas.DataFrame

    Returns
    -------
    prediction : int
    probability : float
    """

    model = load_model()

    processed_input = preprocess_input(input_df)

    prediction = model.predict(processed_input)[0]

    probability = None

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(processed_input)[0][1]

    return prediction, probability


# =====================================================
# Predict Sample from Test Dataset
# =====================================================

def predict_sample():

    model = load_model()

    _, X_test, _, y_test, _ = get_processed_data()

    prediction = model.predict([X_test[0]])[0]

    actual = y_test.iloc[0]

    print("=" * 50)
    print("Sample Prediction")
    print("=" * 50)

    print(f"Predicted Attrition : {'Yes' if prediction else 'No'}")
    print(f"Actual Attrition    : {'Yes' if actual else 'No'}")


if __name__ == "__main__":

    predict_sample()