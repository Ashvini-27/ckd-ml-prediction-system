import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# MODEL PERFORMANCE
# -----------------------------

models = {
    "Logistic Regression": {
        "Accuracy": 0.9975,
        "Precision": 1.0000,
        "Recall": 0.9960,
        "F1": 0.9980,
    },
    "Random Forest": {
        "Accuracy": 0.9925,
        "Precision": 0.9922,
        "Recall": 0.9960,
        "F1": 0.9940,
    },
    "SVM": {
        "Accuracy": 0.9925,
        "Precision": 1.0000,
        "Recall": 0.9880,
        "F1": 0.9939,
    },
    "XGBoost": {
        "Accuracy": 0.9875,
        "Precision": 0.9884,
        "Recall": 0.9920,
        "F1": 0.9900,
    }
}

df = pd.DataFrame(models).T

print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)
print(df.round(4))


# -----------------------------
# SAVE TABLE
# -----------------------------

df.to_csv(
    "results/model_comparison.csv"
)


# -----------------------------
# MODEL COMPARISON GRAPH
# -----------------------------

ax = df[["Accuracy", "Precision", "Recall", "F1"]].plot(
    kind="bar",
    figsize=(10, 6)
)

ax.set_title("Model Performance Comparison")
ax.set_ylabel("Score")
ax.set_ylim(0.90, 1.01)
ax.set_xlabel("Model")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "results/model_comparison.png",
    dpi=300
)

plt.close()


# -----------------------------
# ABLATION RESULTS
# -----------------------------

ablation = pd.DataFrame({
    "Experiment": [
        "All 24 Features",
        "Reduced 14 Features"
    ],
    "Accuracy": [
        0.9975,
        0.8875
    ],
    "Recall": [
        0.9960,
        0.8720
    ],
    "F1": [
        0.9980,
        0.9064
    ]
})

print("\n" + "=" * 70)
print("ABLATION RESULTS")
print("=" * 70)
print(ablation.round(4))

ablation.to_csv(
    "results/ablation_results.csv",
    index=False
)


# -----------------------------
# ABLATION GRAPH
# -----------------------------

ax = ablation.set_index(
    "Experiment"
)[["Accuracy", "Recall", "F1"]].plot(
    kind="bar",
    figsize=(8, 6)
)

ax.set_title("Ablation Analysis")
ax.set_ylabel("Score")
ax.set_ylim(0.80, 1.01)
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "results/ablation_analysis.png",
    dpi=300
)

plt.close()


# -----------------------------
# LEAVE-ONE-OUT RESULTS
# -----------------------------

loo = pd.DataFrame({
    "Feature Removed": [
        "hemo", "pcv", "rbcc", "sg", "sc",
        "al", "htn", "dm", "pe", "ane"
    ],
    "Accuracy": [
        0.9825, 0.9875, 0.9925, 0.9800, 0.9925,
        0.9900, 0.9900, 0.9950, 0.9925, 0.9975
    ]
})

loo = loo.sort_values(
    "Accuracy"
)

print("\n" + "=" * 70)
print("LEAVE-ONE-FEATURE-OUT RESULTS")
print("=" * 70)
print(loo.round(4))

loo.to_csv(
    "results/leave_one_out_results.csv",
    index=False
)


# -----------------------------
# FEATURE IMPACT GRAPH
# -----------------------------

ax = loo.plot(
    x="Feature Removed",
    y="Accuracy",
    kind="bar",
    figsize=(10, 6),
    legend=False
)

ax.set_title("Impact of Removing Individual Features")
ax.set_ylabel("5-Fold CV Accuracy")
ax.set_xlabel("Feature Removed")
ax.set_ylim(0.95, 1.005)

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "results/leave_one_out.png",
    dpi=300
)

plt.close()


print("\nResults saved in the 'results' folder.")