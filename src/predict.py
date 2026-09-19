import sys
from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


# ============================================================
# LOAD MODEL AND PREPROCESSOR
# ============================================================

model = joblib.load(
    MODEL_DIR / "logistic_regression.pkl"
)

preprocessor = joblib.load(
    MODEL_DIR / "preprocessor.pkl"
)


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

numerical_columns = [
    "age",
    "bp",
    "sg",
    "al",
    "su",
    "bgr",
    "bu",
    "sc",
    "sod",
    "pot",
    "hemo",
    "pcv",
    "wbcc",
    "rbcc"
]

categorical_columns = [
    "rbc",
    "pc",
    "pcc",
    "ba",
    "htn",
    "dm",
    "cad",
    "appet",
    "pe",
    "ane"
]


# ============================================================
# GET PATIENT INPUT
# ============================================================

print("\n" + "=" * 60)
print("CHRONIC KIDNEY DISEASE PREDICTION")
print("=" * 60)

print("\nEnter patient information.")
print("Press Enter for an unknown/missing value.\n")


patient = {}


# Numerical features
for column in numerical_columns:

    value = input(f"{column}: ").strip()

    if value == "":
        patient[column] = pd.NA
    else:
        patient[column] = float(value)


# Categorical features
for column in categorical_columns:

    value = input(f"{column}: ").strip().lower()

    if value == "":
        patient[column] = pd.NA
    else:
        patient[column] = value


# ============================================================
# CREATE DATAFRAME
# ============================================================

patient_df = pd.DataFrame(
    [patient]
)


# ============================================================
# PREPROCESS PATIENT DATA
# ============================================================

patient_processed = preprocessor.transform(
    patient_df
)


# ============================================================
# MAKE PREDICTION
# ============================================================

prediction = model.predict(
    patient_processed
)[0]

probability = model.predict_proba(
    patient_processed
)[0][1]


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

if prediction == 1:
    print("Prediction: CKD")
else:
    print("Prediction: NOT CKD")

print(f"CKD probability: {probability:.2%}")

print("=" * 60)