import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

from src.preprocess_data import load_data


df = load_data()

# Encode target
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

# Strongly separating features identified from previous analysis
reduced_features = [
    feature for feature in all_features
    if feature not in [
        "hemo", "pcv", "rbcc", "sg", "sc", "al",
        "htn", "dm", "pe", "ane"
    ]
]


def run_experiment(features, name):

    X = df[features]
    y = df["class_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    numerical_columns = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

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

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(f"Number of features: {len(features)}")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions):.4f}")
    print(f"Recall:    {recall_score(y_test, predictions):.4f}")
    print(f"F1 Score:  {f1_score(y_test, predictions):.4f}")


run_experiment(
    all_features,
    "EXPERIMENT A: ALL 24 FEATURES"
)

run_experiment(
    reduced_features,
    "EXPERIMENT B: WITHOUT STRONGLY SEPARATING FEATURES"
)