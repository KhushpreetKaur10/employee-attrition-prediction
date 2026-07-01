import os

# =====================================================
# Base Directory
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =====================================================
# Data Paths
# =====================================================

DATA_DIR = os.path.join(BASE_DIR, "data")

DATA_PATH = os.path.join(
    DATA_DIR,
    "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

# =====================================================
# Model Paths
# =====================================================

MODEL_DIR = os.path.join(BASE_DIR, "models")

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_model.pkl"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)

FEATURE_COLUMNS_PATH = os.path.join(
    MODEL_DIR,
    "feature_columns.pkl"
)

# =====================================================
# Images
# =====================================================

IMAGES_DIR = os.path.join(BASE_DIR, "images")

# =====================================================
# Train/Test Configuration
# =====================================================

TEST_SIZE = 0.20

RANDOM_STATE = 42

# =====================================================
# Target Column
# =====================================================

TARGET_COLUMN = "Attrition"

# =====================================================
# Columns to Drop
# =====================================================

DROP_COLUMNS = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]

# =====================================================
# Binary Columns
# =====================================================

BINARY_COLUMNS = {
    "Gender": {
        "Male": 0,
        "Female": 1
    },
    "OverTime": {
        "No": 0,
        "Yes": 1
    }
}

# =====================================================
# One-Hot Encoding Columns
# =====================================================

CATEGORICAL_COLUMNS = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "JobRole",
    "MaritalStatus"
]
