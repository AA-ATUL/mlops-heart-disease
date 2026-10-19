
from pathlib import Path
import pandas as pd

# Locate the project and load the downloaded CSV
project_root = Path(__file__).resolve().parent.parent
data_file = project_root / "data" / "raw" / "heart_disease.csv"

df = pd.read_csv(data_file)

print("\n--- 1. Dataset dimensions ---")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n--- 2. Column names and data types ---")
df.info()

print("\n--- 3. Missing values by column ---")
print(df.isna().sum())

print("\n--- 4. Total missing values ---")
print(df.isna().sum().sum())

print("\n--- 5. Target distribution (original num values) ---")
print(df["num"].value_counts(dropna=False).sort_index())

print("\n--- 6. Duplicate rows ---")
print(df.duplicated().sum())

print("\n--- 7. Numerical summary ---")
print(df.describe().round(2))
