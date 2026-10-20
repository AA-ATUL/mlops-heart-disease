
import json
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_saved_model_is_loadable():
    model_path = PROJECT_ROOT / "models" / "best_model.joblib"

    assert model_path.exists()

    model = joblib.load(model_path)

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_model_comparison_contains_both_classifiers():
    comparison_path = PROJECT_ROOT / "reports" / "model_comparison.csv"

    assert comparison_path.exists()

    comparison = pd.read_csv(comparison_path)

    assert set(comparison["model"]) == {
        "logistic_regression",
        "random_forest",
    }
    assert comparison["best_cv_roc_auc"].notna().all()
    assert comparison["best_cv_roc_auc"].between(0, 1).all()


def test_evaluation_metrics_are_valid():
    metrics_path = PROJECT_ROOT / "reports" / "test_metrics.json"

    assert metrics_path.exists()

    with open(metrics_path, encoding="utf-8") as file:
        metrics = json.load(file)

    assert metrics["selected_model"] in {
        "logistic_regression",
        "random_forest",
    }

    for metric in (
        "best_cv_roc_auc",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_roc_auc",
    ):
        assert 0 <= metrics[metric] <= 1


def test_evaluation_plots_exist():
    figure_dir = PROJECT_ROOT / "reports" / "figures"

    assert (figure_dir / "confusion_matrix.png").exists()
    assert (figure_dir / "roc_curve.png").exists()
