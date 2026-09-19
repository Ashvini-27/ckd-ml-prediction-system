import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.preprocess_data import load_and_preprocess_data

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. LOAD DATA
# ============================================================

(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
) = load_and_preprocess_data()


# ============================================================
# 2. TRAIN RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# 3. GET FEATURE IMPORTANCE
# ============================================================

feature_names = X_train.columns

importance_df = pd.DataFrame({
    "encoded_feature": feature_names,
    "importance": model.feature_importances_
})


# ============================================================
# 4. MAP ENCODED FEATURES BACK TO ORIGINAL FEATURES
# ============================================================

def get_original_feature(encoded_feature):

    # Remove transformer prefix
    feature = encoded_feature.split("__", 1)[-1]

    # Numerical features
    if feature in [
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
    ]:
        return feature

    # Categorical features
    categorical_features = [
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

    for categorical_feature in categorical_features:

        if feature.startswith(
            categorical_feature + "_"
        ):
            return categorical_feature

    return feature


importance_df["original_feature"] = (
    importance_df["encoded_feature"]
    .apply(get_original_feature)
)


# ============================================================
# 5. AGGREGATE IMPORTANCE
# ============================================================

aggregated = (
    importance_df
    .groupby("original_feature")["importance"]
    .sum()
    .sort_values(
        ascending=False
    )
)


# ============================================================
# 6. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("AGGREGATED RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)

print(
    aggregated.to_string()
)


# ============================================================
# 7. SAVE RESULTS
# ============================================================

output_path = (
    PROJECT_ROOT
    / "results"
    / "aggregated_feature_importance.csv"
)

aggregated.to_csv(
    output_path,
    header=["importance"]
)

print("\nResults saved to:")
print(output_path)