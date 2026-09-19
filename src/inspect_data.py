import pandas as pd

# Read the ARFF file as plain text
with open("data/chronic_kidney_disease.arff", "r") as file:
    lines = file.readlines()

# Find the @data section
data_start = next(
    i for i, line in enumerate(lines)
    if line.strip().lower() == "@data"
)

# Get column names from the ARFF header
column_names = []

for line in lines[:data_start]:
    line = line.strip()

    if line.lower().startswith("@attribute"):
        parts = line.split()
        column_names.append(parts[1].strip("'"))

print("Number of columns:", len(column_names))

# Read the data
cleaned_data = []

for line_number, line in enumerate(lines[data_start + 1:], start=data_start + 2):

    if not line.strip():
        continue

    values = [value.strip() for value in line.strip().split(",")]

    # Fix malformed rows with an extra empty value at the end
    if len(values) == 26 and values[-1] == "":
        values = values[:-1]

    # Fix the other malformed row containing an extra empty value
    elif len(values) == 26:
        values.pop(19)

    # Check that every row has exactly 25 columns
    if len(values) != len(column_names):
        print(
            f"Problem at line {line_number}: "
            f"{len(values)} values instead of {len(column_names)}"
        )
        continue

    cleaned_data.append(values)

# Create DataFrame
df = pd.DataFrame(cleaned_data, columns=column_names)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())