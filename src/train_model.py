
from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1. Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "heart_disease.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load raw data and create the binary target
df = pd.read_csv(RAW_FILE)

y = (df["num"] > 0).astype(int)
X = df.drop(columns=["num"])

print("Dataset shape:", df.shape)
print("Class distribution:")
print(y.value_counts().sort_index())


# 3. Define numerical and categorical features
numerical_features = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

categorical_features = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "ca", "thal"
]


# 4. Build preprocessing pipelines
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features),
])


# 5. Split data before fitting any preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining records:", len(X_train))
print("Test records:", len(X_test))


# 6. Configure five-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# 7. Define two classifiers and small tuning grids
model_configs = {
    "logistic_regression": {
        "classifier": LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
        "param_grid": {
            "classifier__C": [0.1, 1.0, 10.0],
        },
    },
    "random_forest": {
        "classifier": RandomForestClassifier(
            random_state=42,
        ),
        "param_grid": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 5, 10],
        },
    },
}


# 8. Tune both models using training data only
searches = {}
cv_results_list = []
comparison_rows = []

for model_name, config in model_configs.items():
    print(f"\nTraining and tuning: {model_name}")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", config["classifier"]),
    ])

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=config["param_grid"],
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X_train, y_train)
    searches[model_name] = search

    results = pd.DataFrame(search.cv_results_)
    results["model"] = model_name
    cv_results_list.append(results)

    comparison_rows.append({
        "model": model_name,
        "best_cv_roc_auc": float(search.best_score_),
        "best_params": json.dumps(search.best_params_),
    })

    print("Best CV ROC-AUC:", round(search.best_score_, 4))
    print("Best parameters:", search.best_params_)


# 9. Save all candidate CV results and model comparison
all_cv_results = pd.concat(cv_results_list, ignore_index=True)
all_cv_results.to_csv(REPORT_DIR / "cv_results.csv", index=False)

comparison = pd.DataFrame(comparison_rows)
comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

print("\nModel comparison:")
print(comparison.to_string(index=False))


# 10. Select the winner using cross-validation, not test results
best_name = max(
    searches,
    key=lambda name: searches[name].best_score_,
)
best_search = searches[best_name]
best_model = best_search.best_estimator_

print("\nSelected model:", best_name)


# 11. Evaluate the selected model on the untouched test set
y_pred = best_model.predict(X_test)
y_probability = best_model.predict_proba(X_test)[:, 1]

test_metrics = {
    "selected_model": best_name,
    "best_cv_roc_auc": float(best_search.best_score_),
    "test_accuracy": float(accuracy_score(y_test, y_pred)),
    "test_precision": float(
        precision_score(y_test, y_pred, zero_division=0)
    ),
    "test_recall": float(
        recall_score(y_test, y_pred, zero_division=0)
    ),
    "test_roc_auc": float(roc_auc_score(y_test, y_probability)),
    "best_params": best_search.best_params_,
}

print("\nFinal test metrics:")
for metric, value in test_metrics.items():
    print(f"{metric}: {value}")

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

with open(REPORT_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
    json.dump(test_metrics, f, indent=4)


# 12. Save evaluation plots
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "confusion_matrix.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()

RocCurveDisplay.from_predictions(y_test, y_probability)
plt.title(f"ROC Curve - {best_name}")
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "roc_curve.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()


# 13. Save the complete fitted pipeline
model_path = MODEL_DIR / "best_model.joblib"
joblib.dump(best_model, model_path)

print("\nSaved model:", model_path)
print("Saved CV results:", REPORT_DIR / "cv_results.csv")
print("Saved comparison:", REPORT_DIR / "model_comparison.csv")
print("Saved test metrics:", REPORT_DIR / "test_metrics.json")
print("Saved evaluation plots:", FIGURE_DIR)
