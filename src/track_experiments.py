
from pathlib import Path
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
MODEL_DIR = PROJECT_ROOT / "models"

# Store tracking data inside the project directory.
TRACKING_DB = PROJECT_ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB.as_posix()}")
mlflow.set_experiment("Heart Disease Classification")

# Record every hyperparameter candidate evaluated by GridSearchCV.
cv_results = pd.read_csv(REPORT_DIR / "cv_results.csv")

for _, row in cv_results.iterrows():
    import ast
    params = ast.literal_eval(row["params"])

    with mlflow.start_run(run_name=f'{row["model"]}_candidate'):
        mlflow.log_param("model", row["model"])

        for key, value in params.items():
            if value is None:
                mlflow.log_param(key, "None")
            else:
                mlflow.log_param(key, value)

        mlflow.log_metric(
            "mean_cv_roc_auc",
            float(row["mean_test_score"]),
        )
        mlflow.log_metric(
            "std_cv_roc_auc",
            float(row["std_test_score"]),
        )

        mlflow.set_tag("stage", "cross_validation")
        mlflow.set_tag("source", "GridSearchCV")

# Record the final selected model and its held-out test metrics.
with open(REPORT_DIR / "test_metrics.json", encoding="utf-8") as f:
    metrics = json.load(f)

with mlflow.start_run(run_name="selected_model_test_evaluation"):
    mlflow.log_param("selected_model", metrics["selected_model"])

    for key, value in metrics["best_params"].items():
        mlflow.log_param(key, value)

    for key in [
        "best_cv_roc_auc",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_roc_auc",
    ]:
        mlflow.log_metric(key, float(metrics[key]))

    mlflow.set_tag("stage", "held_out_test_evaluation")

    for filename in [
        "model_comparison.csv",
        "cv_results.csv",
        "test_metrics.json",
    ]:
        mlflow.log_artifact(str(REPORT_DIR / filename))

    for filename in [
        "feature_histograms.png",
        "correlation_heatmap.png",
        "class_balance.png",
        "confusion_matrix.png",
        "roc_curve.png",
    ]:
        path = FIGURE_DIR / filename
        if path.exists():
            mlflow.log_artifact(str(path))

    mlflow.log_artifact(str(MODEL_DIR / "best_model.joblib"))

print("MLflow experiment tracking completed.")
print("Tracking database:", TRACKING_DB)
