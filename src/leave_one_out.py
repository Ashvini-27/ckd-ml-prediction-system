import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

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

strong_features = [
    "hemo", "pcv", "rbcc", "sg", "sc",
    "al", "htn", "dm", "pe", "ane"
]

X = df[all_features]
y = df["class_encoded"]

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


def evaluate(features):

    X_subset = X[features]

    numerical_columns = X_subset.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = X_subset.select_dtypes(
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

    results = cross_validate(
        model,
        X_subset,
        y,
        cv=cv,
        scoring=["accuracy", "recall", "f1"]
    )

    return (
        results["test_accuracy"].mean(),
        results["test_recall"].mean(),
        results["test_f1"].mean()
    )


print("=" * 70)
print("LEAVE-ONE-FEATURE-OUT ANALYSIS")
print("=" * 70)

baseline = evaluate(all_features)

print(
    f"\nBaseline (all features): "
    f"Accuracy={baseline[0]:.4f}, "
    f"Recall={baseline[1]:.4f}, "
    f"F1={baseline[2]:.4f}"
)

print("\nRemoving one feature at a time:\n")

for feature in strong_features:

    features_without = [
        f for f in all_features
        if f != feature
    ]

    result = evaluate(features_without)

    print(
        f"Remove {feature:5s} | "
        f"Accuracy: {result[0]:.4f} | "
        f"Recall: {result[1]:.4f} | "
        f"F1: {result[2]:.4f}"
    )