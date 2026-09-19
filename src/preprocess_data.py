import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ============================================================
# FEATURE DEFINITIONS
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
# LOAD AND CLEAN DATA
# ============================================================

def load_data():

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

                values = [
                    value.strip()
                    for value in line.split(",")
                ]

                # Normal row
                if len(values) == 25:
                    rows.append(values)

                # Malformed row
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

                    else:
                        print("Unresolved malformed row:")
                        print(values)

                else:
                    print("Unresolved row:")
                    print(values)

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

    df = pd.DataFrame(
        rows,
        columns=columns
    )

    # Replace ? with missing values
    df = df.replace("?", pd.NA)

    # Remove unnecessary whitespace
    df = df.map(
        lambda x: x.strip()
        if isinstance(x, str)
        else x
    )

    # Convert numerical columns
    for column in numerical_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df


# ============================================================
# PREPROCESS DATA
# ============================================================

def load_and_preprocess_data():

    df = load_data()

    # Separate features and target
    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column].map({
        "notckd": 0,
        "ckd": 1
    })

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Numerical pipeline
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # Categorical pipeline
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    # Combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ]
    )

    # Fit ONLY on training data
    X_train_processed = (
        preprocessor.fit_transform(X_train)
    )

    # Transform test data using fitted preprocessing
    X_test_processed = (
        preprocessor.transform(X_test)
    )

    # Get feature names
    feature_names = (
        preprocessor.get_feature_names_out()
    )

    # Convert to DataFrames
    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index
    )

    X_test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
        index=X_test.index
    )

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    )


# ============================================================
# RUN PREPROCESSING DIRECTLY
# ============================================================

if __name__ == "__main__":

    (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    ) = load_and_preprocess_data()

    print("=" * 60)
    print("CKD PREPROCESSING PIPELINE")
    print("=" * 60)

    print("\nOriginal Dataset:")
    print(load_data().shape)

    print("\nTraining Set:")
    print(X_train_processed.shape)

    print("\nTesting Set:")
    print(X_test_processed.shape)

    print("\nTraining Target Distribution:")
    print(y_train.value_counts())

    print("\nTesting Target Distribution:")
    print(y_test.value_counts())

    print("\nMissing Values After Preprocessing:")

    print(
        "Training:",
        X_train_processed.isna().sum().sum()
    )

    print(
        "Testing:",
        X_test_processed.isna().sum().sum()
    )

    print("\nFirst 5 Processed Rows:")
    print(X_train_processed.head())

    print("\nPreprocessing complete.")