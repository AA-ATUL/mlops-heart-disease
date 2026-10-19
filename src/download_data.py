
from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Retrieve Heart Disease dataset from UCI (dataset ID 45)
dataset = fetch_ucirepo(id=45)

# Input features and target
X = dataset.data.features
y = dataset.data.targets

print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("Target columns:", y.columns.tolist())
print("Missing values:", X.isna().sum().sum())

# Keep the original target values in the raw dataset
raw_data = pd.concat([X, y], axis=1)

# Save to a path relative to this script's project directory
project_root = Path(__file__).resolve().parent.parent
output_dir = project_root / "data" / "raw"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "heart_disease.csv"
raw_data.to_csv(output_file, index=False)

print("Saved dataset to:", output_file)
print("Dataset shape:", raw_data.shape)
print(raw_data.head())
