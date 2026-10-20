
import ast
import json
from pathlib import Path

import mlflow
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
MODEL_DIR = PROJECT_ROOT / "models"

# Store tracking data inside the project directory.
TRACKING_DB = PROJECT_ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB.as_posix()}")
mlflow.set_experiment("Heart Disease Classification")

# Load cross-validation results from GridSearchCV.
cv_results = pd.read_csv(REPORT_DIR / "cv_results.csv")

# Record every hyperparameter candidate evaluated by GridSearchCV.
for _, row in cv_results.iterrows():
    params = ast.literal_eval(row["params"])

    with mlflow.start_run(
        run_name=f'{row["model"]}_candidate'
    ):
        mlflow.log_param("model", row["model"])

        for key, value in params.items():
            mlflow.log_param(
                key,
                "None" if value is None else value,
            )

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

        # Log shared report artifacts for every candidate run.
        for filename in [
            "model_comparison.csv",
            "cv_results.csv",
        ]:
            path = REPORT_DIR / filename
            if path.exists():
                mlflow.log_artifact(str(path))

        # Log EDA plots for every candidate run.
        for filename in [
            "feature_histograms.png",
            "correlation_heatmap.png",
            "class_balance.png",
        ]:
            path = FIGURE_DIR / filename
            if path.exists():
                mlflow.log_artifact(str(path))

# Load the selected model's held-out test metrics.
with open(
    REPORT_DIR / "test_metrics.json",
    encoding="utf-8",
) as f:
    metrics = json.load(f)

# Record the final selected model and test evaluation.
with mlflow.start_run(
    run_name="selected_model_test_evaluation"
):
    mlflow.log_param(
        "selected_model",
        metrics["selected_model"],
    )

    for key, value in metrics["best_params"].items():
        mlflow.log_param(
            key,
            "None" if value is None else value,
        )

    for key in [
        "best_cv_roc_auc",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_roc_auc",
    ]:
        mlflow.log_metric(key, float(metrics[key]))

    mlflow.set_tag("stage", "held_out_test_evaluation")

    # Log evaluation reports.
    for filename in [
        "model_comparison.csv",
        "cv_results.csv",
        "test_metrics.json",
    ]:
        path = REPORT_DIR / filename
        if path.exists():
            mlflow.log_artifact(str(path))

    # Log EDA and model evaluation plots.
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

    # Log the selected model artifact.
    model_path = MODEL_DIR / "best_model.joblib"
    if model_path.exists():
        mlflow.log_artifact(str(model_path))

print("MLflow experiment tracking completed.")
print("Tracking database:", TRACKING_DB)
