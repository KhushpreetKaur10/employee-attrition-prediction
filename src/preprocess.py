import joblib
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    DATA_PATH,
    TEST_SIZE,
    RANDOM_STATE,
    TARGET_COLUMN,
    DROP_COLUMNS,
    BINARY_COLUMNS,
    CATEGORICAL_COLUMNS,
    SCALER_PATH,
    FEATURE_COLUMNS_PATH,
)


# =====================================================
# Load Dataset
# =====================================================

def load_data():
    """
    Load the employee attrition dataset.
    """
    return pd.read_csv(DATA_PATH)


# =====================================================
# Data Preprocessing
# =====================================================

def preprocess_data(df):
    """
    Clean and preprocess the dataset.
    """

    df = df.copy()

    # Drop unnecessary columns
    df.drop(columns=DROP_COLUMNS, inplace=True)

    # Encode target column
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({
        "No": 0,
        "Yes": 1
    })

    # Binary Encoding
    for column, mapping in BINARY_COLUMNS.items():
        df[column] = df[column].map(mapping)

    # One-Hot Encoding
    df = pd.get_dummies(
        df,
        columns=CATEGORICAL_COLUMNS,
        drop_first=True,
        dtype=int
    )

    return df


# =====================================================
# Train Test Split
# =====================================================

def split_data(df):
    """
    Split data into train and test sets.
    """

    X = df.drop(TARGET_COLUMN, axis=1)

    y = df[TARGET_COLUMN]

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y

    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    joblib.dump(
        scaler,
        SCALER_PATH
    )

    # Save feature names
    joblib.dump(
        feature_names,
        FEATURE_COLUMNS_PATH
    )

    return (

        X_train_scaled,

        X_test_scaled,

        y_train,

        y_test,

        feature_names

    )


# =====================================================
# Complete Pipeline
# =====================================================

def get_processed_data():
    """
    Complete preprocessing pipeline.
    """

    df = load_data()

    df = preprocess_data(df)

    return split_data(df)


# =====================================================
# Preprocess User Input
# =====================================================

def preprocess_input(input_df):
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            "Scaler not found. Please run 'python -m src.train' first."
        )

    if not os.path.exists(FEATURE_COLUMNS_PATH):
        raise FileNotFoundError(
            "Feature columns not found. Please run 'python -m src.train' first."
        )
    """
    Preprocess new employee data for prediction.
    """

    input_df = input_df.copy()

    # Drop unwanted columns if present
    for column in DROP_COLUMNS:
        if column in input_df.columns:
            input_df.drop(column=column, inplace=True)

    # Binary Encoding
    for column, mapping in BINARY_COLUMNS.items():
        if column in input_df.columns:
            input_df[column] = input_df[column].map(mapping)

    # One-Hot Encoding
    input_df = pd.get_dummies(

        input_df,

        columns=CATEGORICAL_COLUMNS,

        drop_first=True,

        dtype=int

    )

    # Load feature names used during training
    feature_columns = joblib.load(
        FEATURE_COLUMNS_PATH
    )

    # Add missing columns
    for column in feature_columns:

        if column not in input_df.columns:

            input_df[column] = 0

    # Keep only training columns
    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Convert all columns to numeric
    input_df = input_df.astype(float)

    # Load scaler
    scaler = joblib.load(
        SCALER_PATH
    )

    input_scaled = scaler.transform(input_df)

    return input_scaled