import pandas as pd

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

            # Handle malformed rows with 26 values
            elif len(values) == 26:

                # First two malformed rows have an extra empty
                # value at the end
                if values[-1] == "":
                    values = values[:-1]
                    rows.append(values)

                # Third malformed row has an extra "no" after
                # the missing dm value
                elif (
                    values[19] == ""
                    and values[20] == "no"
                    and values[-1] == "notckd"
                ):
                    values.pop(20)
                    rows.append(values)

                else:
                    print("\nUNRESOLVED MALFORMED ROW")
                    print(values)

            else:
                print("\nUNRESOLVED MALFORMED ROW")
                print("Number of values:", len(values))
                print(values)


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


# ============================================================
# 4. REPLACE MISSING VALUES
# ============================================================

df = df.replace("?", pd.NA)


# ============================================================
# 5. REMOVE EXTRA WHITESPACE
# ============================================================

df = df.map(
    lambda x: x.strip() if isinstance(x, str) else x
)


# ============================================================
# 6. DEFINE FEATURE TYPES
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

target_column = "class"


# ============================================================
# 7. CONVERT NUMERICAL COLUMNS
# ============================================================

for column in numerical_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# 8. BASIC DATASET INFORMATION
# ============================================================

print("=" * 60)
print("CKD DATASET ANALYSIS")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nNumber of Columns:")
print(len(df.columns))


# ============================================================
# 9. DATA TYPES
# ============================================================

print("\nData Types:")
print(df.dtypes)


# ============================================================
# 10. MISSING VALUES
# ============================================================

print("\nMissing Values:")
print(df.isna().sum())


print("\nMissing Value Percentage:")
print(
    (df.isna().mean() * 100).round(2)
)


# ============================================================
# 11. TARGET DISTRIBUTION
# ============================================================

print("\nTarget Distribution:")
print(df[target_column].value_counts())


print("\nTarget Distribution (%):")
print(
    (df[target_column].value_counts(normalize=True) * 100).round(2)
)


# ============================================================
# 12. NUMERICAL DATA VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("NUMERICAL DATA VALIDATION")
print("=" * 60)

print("\nNumerical Data Types:")

print(
    df[numerical_columns].dtypes
)


print("\nMissing Values in Numerical Columns:")

for column in numerical_columns:
    missing_count = df[column].isna().sum()

    print(
        f"{column}: {missing_count}"
    )


# ============================================================
# 13. CATEGORICAL DATA VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("CATEGORICAL DATA VALIDATION")
print("=" * 60)

for column in categorical_columns:

    print(f"\n{column}:")

    print(
        df[column].value_counts(dropna=False)
    )


# ============================================================
# 14. CHECK FOR UNEXPECTED CATEGORICAL VALUES
# ============================================================

expected_categories = {
    "rbc": ["normal", "abnormal"],
    "pc": ["normal", "abnormal"],
    "pcc": ["present", "notpresent"],
    "ba": ["present", "notpresent"],
    "htn": ["yes", "no"],
    "dm": ["yes", "no"],
    "cad": ["yes", "no"],
    "appet": ["good", "poor"],
    "pe": ["yes", "no"],
    "ane": ["yes", "no"]
}

print("\n" + "=" * 60)
print("UNEXPECTED CATEGORICAL VALUES")
print("=" * 60)

for column in categorical_columns:

    valid_values = expected_categories[column]

    unexpected_values = df[
        ~df[column].isin(valid_values)
        & df[column].notna()
    ][column].unique()

    if len(unexpected_values) == 0:
        print(f"{column}: None")
    else:
        print(
            f"{column}: {unexpected_values}"
        )


# ============================================================
# 15. CHECK TARGET VALUES
# ============================================================

print("\n" + "=" * 60)
print("TARGET VALIDATION")
print("=" * 60)

expected_targets = ["ckd", "notckd"]

unexpected_targets = df[
    ~df[target_column].isin(expected_targets)
][target_column].unique()

if len(unexpected_targets) == 0:
    print("Target values: Valid")
else:
    print(
        "Unexpected target values:",
        unexpected_targets
    )


# ============================================================
# 16. FINAL DATASET CHECK
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATASET CHECK")
print("=" * 60)

print(
    f"\nRows: {df.shape[0]}"
)

print(
    f"Columns: {df.shape[1]}"
)

print(
    f"Numerical features: {len(numerical_columns)}"
)

print(
    f"Categorical features: {len(categorical_columns)}"
)

print(
    f"Target column: {target_column}"
)

print("\nAnalysis complete.")