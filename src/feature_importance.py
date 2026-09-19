import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt

from src.preprocess_data import load_and_preprocess_data

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. LOAD PREPROCESSED DATA
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

importance = model.feature_importances_

feature_names = X_train.columns


importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
})


# Sort by importance
importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)


# ============================================================
# 4. DISPLAY TOP FEATURES
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)

print(
    importance_df.head(15).to_string(
        index=False
    )
)


# ============================================================
# 5. SAVE RESULTS
# ============================================================

output_path = (
    PROJECT_ROOT
    / "results"
    / "random_forest_feature_importance.csv"
)

importance_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# 6. PLOT TOP 15 FEATURES
# ============================================================

top_features = importance_df.head(15)

plt.figure(
    figsize=(10, 7)
)

plt.barh(
    top_features["feature"][::-1],
    top_features["importance"][::-1]
)

plt.xlabel("Feature Importance")

plt.ylabel("Feature")

plt.title(
    "Random Forest Feature Importance"
)

plt.tight_layout()

plot_path = (
    PROJECT_ROOT
    / "results"
    / "random_forest_feature_importance.png"
)

plt.savefig(
    plot_path,
    dpi=300
)

plt.show()


print("\nResults saved to:")
print(output_path)

print("\nPlot saved to:")
print(plot_path)