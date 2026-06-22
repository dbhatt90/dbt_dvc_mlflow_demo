import pandas as pd
import numpy as np
import pickle
import json
import yaml
import mlflow
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# ── load params ───────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

THRESHOLD = params["evaluate"]["threshold"]

# ── load data and model ───────────────────────────────────────
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# ── predict ───────────────────────────────────────────────────
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= THRESHOLD).astype(int)

# ── metrics ───────────────────────────────────────────────────
metrics = {
    "accuracy":  round(accuracy_score(y_test, y_pred), 4),
    "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
    "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
    "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
    "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
}

print("\nEvaluation Results:")
for k, v in metrics.items():
    print(f"  {k}: {v}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["retained", "churned"]))

# ── confusion matrix as csv (for dvc plots) ───────────────────
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame({
    "actual":    y_test.values,
    "predicted": y_pred,
    "probability": y_prob,
})

os.makedirs("plots", exist_ok=True)
cm_df.to_csv("plots/predictions.csv", index=False)

# ── save metrics.json (for dvc metrics) ───────────────────────
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ── log to the same mlflow run train.py opened ────────────────
run_id_file = "models/mlflow_run_id.txt"
if os.path.exists(run_id_file):
    with open(run_id_file) as f:
        run_id = f.read().strip()

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact("plots/predictions.csv")

    print(f"\nMetrics logged to MLflow run: {run_id}")

print("\nEvaluation complete.")