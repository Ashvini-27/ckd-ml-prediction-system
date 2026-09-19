import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split


# ============================================================
# 1. LOAD ARFF DATASET
# ============================================================

file_path = "data/chronic_kidney_disease.arff"

rows = []
data_started = False

with open(file_path, "r") as file:
    for line in file:
        line = line.strip()

        if line.lower() == "@data":
            data_started = True
            continue

        if data_started and line and not line.startswith("%"):

            values = [value.strip() for value in line.split(",")]

            # Normal row
            if len(values) == 25:
                rows.append(values)

            # Handle malformed rows
            elif len(values) == 26:

                # Extra empty value at the end
                if values[-1] == "":
                    values = values[:-1]
                    rows.append(values)

                # Missing dm + extra "no"
                elif (
                    values[19] == ""
                    and values[20] == "no"
                    and values[-1] == "notckd"
                ):
                    values.pop(20)
                    rows.append(values)


# ============================================================
# 2. COLUMN NAMES
# ============================================================

columns = [
    "age",
    "bp",
    "sg",
    "al",
    "su",
    "rbc",
    "pc",
    "pcc",
    "ba",
    "bgr",
    "bu",
    "sc",
    "sod",
    "pot",
    "hemo",
    "pcv",
    "wbcc",
    "rbcc",
    "htn",
    "dm",
    "cad",
    "appet",
    "pe",
    "ane",
    "class"
]


# ============================================================
# 3. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(rows, columns=columns)

df = df.replace("?", pd.NA)

df = df.map(
    lambda x: x.strip() if isinstance(x, str) else x
)


# ============================================================
# 4. NUMERICAL FEATURES
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

for column in numerical_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# 5. ENCODE TARGET
# ============================================================

df["class_encoded"] = df["class"].map({
    "notckd": 0,
    "ckd": 1
})


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["class_encoded"]
)


# ============================================================
# 7. CORRELATION ANALYSIS
# ============================================================

correlation_columns = numerical_columns + ["class_encoded"]

correlation_matrix = train_df[
    correlation_columns
].corr()


# ============================================================
# 8. CORRELATION WITH TARGET
# ============================================================

target_correlations = (
    correlation_matrix["class_encoded"]
    .drop("class_encoded")
    .sort_values(
        key=abs,
        ascending=False
    )
)


print("=" * 60)
print("CKD FEATURE ANALYSIS")
print("=" * 60)

print("\nTraining Dataset Shape:")
print(train_df.shape)

print("\nTesting Dataset Shape:")
print(test_df.shape)

print("\nNumerical Feature Correlation With CKD:")
print(target_correlations.round(3))


# ============================================================
# 9. CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title(
    "Feature Correlation Matrix - Training Data"
)

plt.tight_layout()

plt.show()


# ============================================================
# 10. TOP FEATURES
# ============================================================

print("\n" + "=" * 60)
print("TOP NUMERICAL FEATURES")
print("=" * 60)

print(
    target_correlations.head(10).round(3)
)


print("\nFeature analysis complete.")