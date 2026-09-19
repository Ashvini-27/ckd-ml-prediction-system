import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

from src.preprocess_data import load_data


df = load_data()

df["class_encoded"] = df["class"].map({
    "notckd": 0,
    "ckd": 1
})

all_features = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc",
    "ba", "bgr", "bu", "sc", "sod", "pot", "hemo",
    "pcv", "wbcc", "rbcc", "htn", "dm", "cad", "appet",
    "pe", "ane"
]

reduced_features = [
    feature for feature in all_features
    if feature not in [
        "hemo", "pcv", "rbcc", "sg", "sc", "al",
        "htn", "dm", "pe", "ane"
    ]
]


def run_cv(features, name):

    X = df[features]
    y = df["class_encoded"]

    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=["int64", "float64"]
    ).columns.tolist()

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = [
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring
    )

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    for metric in scoring:
        scores = results[f"test_{metric}"]

        print(
            f"{metric.capitalize():10s}: "
            f"{scores.mean():.4f} +/- {scores.std():.4f}"
        )


run_cv(
    all_features,
    "ALL 24 FEATURES"
)

run_cv(
    reduced_features,
    "REDUCED 14 FEATURES"
)