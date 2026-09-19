from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


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
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="CKD Prediction API"
)


# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# INPUT DATA MODEL
# ============================================================

class PatientData(BaseModel):

    age: float
    bp: float
    sg: float
    al: float
    su: float
    bgr: float
    bu: float
    sc: float
    sod: float
    pot: float
    hemo: float
    pcv: float
    wbcc: float
    rbcc: float

    rbc: str
    pc: str
    pcc: str
    ba: str
    htn: str
    dm: str
    cad: str
    appet: str
    pe: str
    ane: str


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(patient: PatientData):

    patient_dict = patient.model_dump()

    patient_df = pd.DataFrame(
        [patient_dict]
    )

    # Apply the same preprocessing used during training
    patient_processed = preprocessor.transform(
        patient_df
    )

    # Prediction
    prediction = model.predict(
        patient_processed
    )[0]

    probability = model.predict_proba(
        patient_processed
    )[0][1]

    if prediction == 1:
        result = "CKD"
    else:
        result = "NOT CKD"

    return {
        "prediction": result,
        "probability": round(float(probability), 4)
    }


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "CKD Prediction API is running"
    }