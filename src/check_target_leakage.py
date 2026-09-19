import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.preprocess_data import load_data


# ============================================================
# 1. LOAD DATA
# ============================================================

df = load_data()


# ============================================================
# 2. CATEGORICAL FEATURES
# ============================================================

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
# 3. TARGET DISTRIBUTION BY CATEGORICAL FEATURE
# ============================================================

print("\n" + "=" * 70)
print("TARGET LEAKAGE / FEATURE-TARGET ASSOCIATION CHECK")
print("=" * 70)

for column in categorical_columns:

    print("\n" + "-" * 70)
    print(column)
    print("-" * 70)

    table = pd.crosstab(
        df[column],
        df["class"],
        margins=True
    )

    print(table)


# ============================================================
# 4. UNIQUE TARGET VALUES
# ============================================================

print("\n" + "=" * 70)
print("TARGET VALUES")
print("=" * 70)

print(df["class"].value_counts())


# ============================================================
# 5. CHECK WHETHER ANY FEATURE PERFECTLY SEPARATES TARGET
# ============================================================

print("\n" + "=" * 70)
print("POTENTIAL PERFECT SEPARATION")
print("=" * 70)

for column in categorical_columns:

    grouped = (
        df.groupby(column)["class"]
        .nunique()
    )

    for value, unique_targets in grouped.items():

        if unique_targets == 1:

            target_value = (
                df.loc[
                    df[column] == value,
                    "class"
                ].iloc[0]
            )

            print(
                f"{column} = {value} "
                f"-> only {target_value}"
            )