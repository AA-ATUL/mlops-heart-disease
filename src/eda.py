
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Project paths
project_root = Path(__file__).resolve().parent.parent
data_file = project_root / "data" / "processed" / "heart_disease.csv"
figure_dir = project_root / "reports" / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)

# Load the processed data
df = pd.read_csv(data_file)

sns.set_theme(style="whitegrid")

# A. Histograms for selected continuous numerical features
continuous_features = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

df[continuous_features].hist(
    bins=15,
    figsize=(12, 8),
    edgecolor="black"
)
plt.suptitle("Heart Disease Dataset: Feature Distributions")
plt.tight_layout()
plt.savefig(
    figure_dir / "feature_histograms.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

# B. Correlation heatmap for the input features and binary target
correlation_columns = [
    "age", "trestbps", "chol", "thalach", "oldpeak", "target"
]
correlation = df[correlation_columns].corr()

plt.figure(figsize=(9, 7))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(
    figure_dir / "correlation_heatmap.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

# C. Binary target class balance
class_counts = df["target"].value_counts().sort_index()

plt.figure(figsize=(7, 5))
ax = sns.barplot(
    x=class_counts.index.astype(str),
    y=class_counts.values
)
ax.bar_label(ax.containers[0])

plt.xlabel("Target (0 = no disease, 1 = disease present)")
plt.ylabel("Number of records")
plt.title("Heart Disease Target Class Balance")
plt.tight_layout()
plt.savefig(
    figure_dir / "class_balance.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

print("EDA completed.")
print("Figures saved to:", figure_dir)
for filename in [
    "feature_histograms.png",
    "correlation_heatmap.png",
    "class_balance.png"
]:
    print("-", figure_dir / filename)
