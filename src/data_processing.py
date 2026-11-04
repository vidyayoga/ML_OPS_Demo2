from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "telco_customer_churn.csv"
# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
def load_data():
    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE)
    print(f"Dataset shape: {df.shape}")
    return df
# ---------------------------------------------------------
# Prepare features and target
# ---------------------------------------------------------
def prepare_data(df):
    # Remove customer ID because it is not useful
    # for predicting churn
    df = df.drop("customerID", axis=1)
    # Convert TotalCharges to numeric
    # Invalid/blank values become NaN
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )
    # Convert target into 0 and 1
    # No  -> 0
    # Yes -> 1
    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })
    # Separate features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    return X, y
# ---------------------------------------------------------
# Create preprocessing pipeline
# ---------------------------------------------------------
def create_preprocessor(X):
    # Identify numerical columns
    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    # Identify categorical columns
    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns.tolist()
    print("\nNumerical columns:")
    print(numerical_columns)
    print("\nCategorical columns:")
    print(categorical_columns)
    # Numerical preprocessing
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )
    # Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )
    # Combine both preprocessing pipelines
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
    return preprocessor
# ---------------------------------------------------------
# Train-test split
# ---------------------------------------------------------
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print("\nData split completed.")
    print(f"Training data: {X_train.shape}")
    print(f"Testing data: {X_test.shape}")
    return X_train, X_test, y_train, y_test
# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    # Load dataset
    df = load_data()
    # Prepare X and y
    X, y = prepare_data(df)
    print("\nFeatures shape:", X.shape)
    print("Target shape:", y.shape)
    # Create preprocessing pipeline
    preprocessor = create_preprocessor(X)
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    print("\nPreprocessing pipeline created successfully.")