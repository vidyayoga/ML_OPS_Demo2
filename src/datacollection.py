from pathlib import Path
import pandas as pd
# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Data directory
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
# IBM Telco Customer Churn dataset
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
# Local file path
OUTPUT_FILE = DATA_DIR / "telco_customer_churn.csv"
def collect_data():
    print("Downloading Telco Customer Churn dataset...")
    # Load dataset
    df = pd.read_csv(DATA_URL)
    print("\nData downloaded successfully!")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())
    # Save locally
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDataset saved to:")
    print(OUTPUT_FILE)
if __name__ == "__main__":
    collect_data()