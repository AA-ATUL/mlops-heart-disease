
from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer

# 1. Locate the project folders
project_root = Path(__file__).resolve().parent.parent
raw_file = project_root / "data" / "raw" / "heart_disease.csv"
output_dir = project_root / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

# 2. Load the original dataset
df = pd.read_csv(raw_file)

# 3. Create a binary target; preserve the original num column
df["target"] = (df["num"] > 0).astype(int)

# 4. Fill missing values in coded categorical features
columns_to_impute = ["ca", "thal"]
imputer = SimpleImputer(strategy="most_frequent")
df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])

# 5. Save the analysis-ready dataset
output_file = output_dir / "heart_disease.csv"
df.to_csv(output_file, index=False)

# 6. Report what happened
print("Saved:", output_file)
print("Dataset shape:", df.shape)
print("Missing values remaining:", df.isna().sum().sum())
print("\nBinary target counts:")
print(df["target"].value_counts().sort_index())
print("\nOriginal target counts:")
print(df["num"].value_counts().sort_index())
