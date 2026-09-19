import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.preprocess_data import load_data

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier


# ============================================================
# 1. LOAD RAW DATA
# ============================================================

df = load_data()

X = df.drop(columns=["class"])

y = df["class"].map({
    "notckd": 0,
    "ckd": 1
})


# ============================================================
# 2. FEATURE DEFINITIONS
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
# 3. PREPROCESSING PIPELINE
# ============================================================

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ]
)


# ============================================================
# 4. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "SVM": SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )
}


# ============================================================
# 5. 5-FOLD STRATIFIED CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 6. EVALUATION METRICS
# ============================================================

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}


# ============================================================
# 7. RUN CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("5-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 70)

print("\nDataset:")
print(f"Samples : {len(X)}")
print(f"Features: {X.shape[1]}")

print("\nEach fold:")
print("Training samples   : approximately 320")
print("Validation samples : approximately 80")


for model_name, model in models.items():

    print("\n" + "-" * 70)
    print(model_name)
    print("-" * 70)

    # IMPORTANT:
    # Preprocessing is inside the pipeline.
    # Therefore each fold fits preprocessing ONLY
    # on that fold's training data.

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )

    accuracy_mean = results["test_accuracy"].mean()
    accuracy_std = results["test_accuracy"].std()

    precision_mean = results["test_precision"].mean()
    precision_std = results["test_precision"].std()

    recall_mean = results["test_recall"].mean()
    recall_std = results["test_recall"].std()

    f1_mean = results["test_f1"].mean()
    f1_std = results["test_f1"].std()

    roc_auc_mean = results["test_roc_auc"].mean()
    roc_auc_std = results["test_roc_auc"].std()

    print(
        f"Accuracy  : {accuracy_mean:.4f} "
        f"+/- {accuracy_std:.4f}"
    )

    print(
        f"Precision : {precision_mean:.4f} "
        f"+/- {precision_std:.4f}"
    )

    print(
        f"Recall    : {recall_mean:.4f} "
        f"+/- {recall_std:.4f}"
    )

    print(
        f"F1-Score  : {f1_mean:.4f} "
        f"+/- {f1_std:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc_mean:.4f} "
        f"+/- {roc_auc_std:.4f}"
    )


print("\n" + "=" * 70)
print("CROSS-VALIDATION COMPLETE")
print("=" * 70)