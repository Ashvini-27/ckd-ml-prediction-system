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
# 2. NUMERICAL FEATURES
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


# ============================================================
# 3. SUMMARY BY TARGET
# ============================================================

print("\n" + "=" * 70)
print("NUMERICAL FEATURE DISTRIBUTION BY TARGET")
print("=" * 70)

for column in numerical_columns:

    print("\n" + "-" * 70)
    print(column)
    print("-" * 70)

    summary = (
        df.groupby("class")[column]
        .agg(
            count="count",
            mean="mean",
            median="median",
            min="min",
            max="max"
        )
    )

    print(summary)


# ============================================================
# 4. RANGE OVERLAP CHECK
# ============================================================

print("\n" + "=" * 70)
print("RANGE OVERLAP CHECK")
print("=" * 70)

for column in numerical_columns:

    ckd_values = df.loc[
        df["class"] == "ckd",
        column
    ].dropna()

    notckd_values = df.loc[
        df["class"] == "notckd",
        column
    ].dropna()

    if len(ckd_values) == 0 or len(notckd_values) == 0:
        continue

    ckd_min = ckd_values.min()
    ckd_max = ckd_values.max()

    notckd_min = notckd_values.min()
    notckd_max = notckd_values.max()

    overlap = not (
        ckd_max < notckd_min
        or notckd_max < ckd_min
    )

    print(
        f"{column:5s} | "
        f"CKD: {ckd_min:.2f} - {ckd_max:.2f} | "
        f"NOT CKD: {notckd_min:.2f} - {notckd_max:.2f} | "
        f"Overlap: {overlap}"
    )