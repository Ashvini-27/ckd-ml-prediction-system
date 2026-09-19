import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD CLEANED DATASET
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
# 4. CLASS DISTRIBUTION
# ============================================================

class_counts = df["class"].value_counts()

class_percentages = (
    df["class"]
    .value_counts(normalize=True)
    * 100
)


# ============================================================
# 5. DISPLAY RESULTS
# ============================================================

print("=" * 60)
print("CKD CLASS BALANCE ANALYSIS")
print("=" * 60)

print("\nClass Counts:")
print(class_counts)

print("\nClass Percentages:")
print(class_percentages.round(2))


# ============================================================
# 6. IMBALANCE RATIO
# ============================================================

majority_class = class_counts.max()
minority_class = class_counts.min()

imbalance_ratio = majority_class / minority_class

print("\nMajority Class:")
print(class_counts.idxmax())

print("\nMinority Class:")
print(class_counts.idxmin())

print("\nImbalance Ratio:")
print(round(imbalance_ratio, 2))


# ============================================================
# 7. VISUALIZATION
# ============================================================

plt.figure(figsize=(7, 5))

class_counts.plot(kind="bar")

plt.title("CKD Dataset Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Patients")

plt.xticks(rotation=0)

plt.tight_layout()

plt.show()


# ============================================================
# 8. CONCLUSION
# ============================================================

print("\n" + "=" * 60)
print("BALANCE ANALYSIS COMPLETE")
print("=" * 60)

print(
    "\nThe dataset shows moderate class imbalance."
)

print(
    "SMOTE will NOT be applied at this stage."
)

print(
    "We will first establish baseline model performance."
)