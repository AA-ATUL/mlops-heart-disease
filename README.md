# Heart Disease Classification --- End-to-End MLOps

**Repository:** https://github.com/AA-ATUL/mlops-heart-disease

This project demonstrates an end-to-end machine-learning workflow for
binary heart-disease classification, from UCI data acquisition and
exploratory analysis to experiment tracking, automated testing, Docker
packaging, Kubernetes deployment, and API monitoring.

> **Important:** This is an educational machine-learning project, not a
> medical diagnostic tool. Predictions must not be used to make clinical
> decisions.

## Architecture

``` text
UCI Heart Disease dataset
        |
        v
Download + preprocessing + EDA
        |
        v
Feature preprocessing pipeline
(imputation, scaling, categorical encoding)
        |
        v
Model comparison + stratified cross-validation
(Logistic Regression and Random Forest)
        |
        +------> MLflow experiment tracking
        |
        v
Selected model saved with joblib
        |
        v
FastAPI service (/health, /predict, /metrics)
        |
        v
Docker image
        |
        v
Kubernetes Deployment + Service + Ingress
```

## Repository layout

``` text
.
├── .github/workflows/ci.yml
├── data/
├── deployment/
├── models/
├── reports/
├── screenshots/
├── src/
│   ├── download_data.py
│   ├── preprocess_data.py
│   ├── eda.py
│   ├── train_model.py
│   ├── track_experiments.py
│   └── api.py
├── tests/
├── Dockerfile
└── requirements.txt
```

## Dataset and preprocessing

The acquisition script downloads the Heart Disease dataset from the UCI
Machine Learning Repository using `ucimlrepo` (dataset ID 45). The
dataset contains 303 records and 14 original columns. The original `num`
target is converted to a binary target: `0` means no detected disease
and values greater than `0` mean disease present.

The initial data check found six missing values: four in `ca` and two in
`thal`; no duplicate rows were found. The target distribution was 164
negative and 139 positive cases. The preprocessing pipeline imputes
numeric values with the median, imputes categorical values with the most
frequent category, scales numeric features, and one-hot encodes
categorical features. The fitted preprocessing and classifier are saved
together as a reusable model pipeline.

## Setup and run

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/download_data.py
python src/preprocess_data.py
python src/eda.py
python src/train_model.py
python src/track_experiments.py
python -m pytest tests -v
ruff check src tests
```

EDA outputs include feature histograms, a correlation heatmap, and a
class-balance plot under `reports/figures/`.

## Model development and evaluation

Two classifiers are compared: Logistic Regression and Random Forest.
Model selection uses a stratified 80/20 train-test split with
`random_state=42` and five-fold stratified cross-validation.
Hyperparameters are tuned using `GridSearchCV` with ROC-AUC as the
selection score.

  Model                                        Mean cross-validation ROC-AUC
  ------------------------------------------ -------------------------------
  Logistic Regression (best configuration)                            0.9025
  Random Forest (best configuration)                                  0.8972

Logistic Regression was selected based on its cross-validation ROC-AUC.

  Held-out test metric     Result
  ---------------------- --------
  Accuracy                 0.8852
  Precision                0.8387
  Recall                   0.9286
  ROC-AUC                  0.9665

These results describe performance on this dataset and split; they do
not establish clinical suitability or generalization to other
populations.

The selected pipeline is saved at `models/best_model.joblib`; evaluation
tables and plots are saved under `reports/`.

## MLflow experiment tracking

The tracking script records candidate model runs and the final
selected-model evaluation. Logged information includes hyperparameters,
cross-validation results, test metrics, comparison tables, EDA figures,
evaluation plots, and the saved model artifact where applicable.

``` powershell
python src/track_experiments.py
mlflow ui
```

Open `http://127.0.0.1:5000`. If a local SQLite tracking database is
configured, ensure the MLflow UI uses the same tracking URI as the
project.

## FastAPI endpoints

-   `GET /health` --- service/model health.
-   `POST /predict` --- returns a prediction, label, confidence, and
    disease probability.
-   `GET /metrics` (or `/metrics/`, depending on routing) ---
    Prometheus-format request and prediction metrics.

Start the API:

``` powershell
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Open interactive API documentation at `http://127.0.0.1:8000/docs`. Send
a JSON request to `/predict` using the exact feature names and valid
values expected by the API schema.

## Docker

``` powershell
docker build -t heart-disease-api:local .
docker run --rm -p 8000:8000 heart-disease-api:local
```

Verify `http://127.0.0.1:8000/health`, then use `/docs` or call
`/predict`. The Dockerfile packages the API and saved model pipeline in
a Python 3.12 slim image.

## GitHub Actions CI

The workflow in `.github/workflows/ci.yml` runs on pushes and pull
requests to `main` or `master`. It sets up Python 3.12, installs
dependencies, runs Ruff linting, downloads and preprocesses data,
generates EDA outputs, trains models, tracks MLflow experiments, runs
pytest, builds the Docker image, and uploads reports and the saved model
as workflow artifacts. The latest workflow run was confirmed green
during project validation.

## Kubernetes deployment

The `deployment/` directory contains Kubernetes Deployment, Service, and
Ingress manifests. The application was tested on a local Minikube
cluster, including health and prediction requests and API metrics.
Screenshots in `screenshots/` document deployed resources and endpoint
checks.

``` bash
kubectl apply -f deployment/deployment.yaml
kubectl apply -f deployment/service.yaml
kubectl apply -f deployment/ingress.yaml
kubectl get deployments,pods,services,ingress
```

Ensure Minikube and the required Ingress controller are running. Image
tags and Ingress addresses depend on the checked-in manifests and local
environment.

## Tests and evidence

The project validation recorded nine passing tests and a successful Ruff
check. Evidence and artifacts are organized as follows:

-   EDA and evaluation outputs: `reports/`
-   Trained model pipeline: `models/best_model.joblib`
-   Deployment and metrics screenshots: `screenshots/`
-   CI workflow: `.github/workflows/ci.yml`
-   Kubernetes manifests: `deployment/`

## Limitations and future work

The project uses a small, historical dataset and a single train/test
split for final evaluation. Further work should include external
validation, calibration, drift monitoring, API authentication and
authorization, security hardening, and clinical governance before any
real-world medical use.
