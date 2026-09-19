import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.preprocess_data import load_and_preprocess_data

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# STEP 1: LOAD PREPROCESSED DATA
# ============================================================

(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
) = load_and_preprocess_data()


# ============================================================
# STEP 2: CREATE XGBOOST MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)


# ============================================================
# STEP 3: TRAIN MODEL
# ============================================================

model.fit(
    X_train,
    y_train
)


# ============================================================
# STEP 4: PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# ============================================================
# STEP 5: EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# STEP 6: DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("XGBOOST BASELINE")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

print("\nFormat:")
print("[[TN, FP]")
print(" [FN, TP]]")


print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["NOT CKD", "CKD"],
        zero_division=0
    )
)