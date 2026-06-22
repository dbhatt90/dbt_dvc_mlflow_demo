import pandas as pd
import numpy as np
import pickle
import yaml
import os
import mlflow
import mlflow.sklearn
import subprocess

# create the models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# load train params
with open("params.yaml") as f:
   params = yaml.safe_load(f)

train_params = params["train"]

MODEL_TYPE = train_params["model"]

# load processed data
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv")

print(f"Training {MODEL_TYPE} on {len(X_train)} samples")
print(f"Features: {list(X_train.columns)}")


# ── build model ───────────────────────────────────────────────
if MODEL_TYPE == "random_forest":
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators = train_params["n_estimators"],
        max_depth    = train_params["max_depth"],
        random_state = train_params["random_state"],
        class_weight = "balanced",   # handles churn class imbalance
        n_jobs       = -1,
    )

elif MODEL_TYPE == "logistic_regression":
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(
        C            = train_params["C"],
        max_iter     = train_params["max_iter"],
        random_state = train_params["random_state"],
        class_weight = "balanced",
        solver       = "lbfgs",
    )

elif MODEL_TYPE == "xgboost":
    from xgboost import XGBClassifier
    model = XGBClassifier(
        learning_rate = train_params["learning_rate"],
        n_estimators  = train_params["n_estimators_xgb"],
        random_state  = train_params["random_state"],
        use_label_encoder = False,
        eval_metric   = "logloss",
        n_jobs        = -1,
    )

else:
    raise ValueError(f"Unknown model type: {MODEL_TYPE}")

# ── train ─────────────────────────────────────────────────────
model.fit(X_train, y_train)
print("Training complete")

# ── mlflow logging ────────────────────────────────────────────
dvc_exp_name = os.environ.get("DVC_EXP_NAME", "not_a_dvc_experiment")
print(f'Experiment name: {dvc_exp_name}')

mlflow.set_experiment("churn-classifier")

with mlflow.start_run():
    # log all params flat
    mlflow.log_param("model_type",   MODEL_TYPE)
    mlflow.log_param("dvc_exp_name",   dvc_exp_name)
    mlflow.log_param("train_size",   len(X_train))
    mlflow.log_params({k: v for k, v in train_params.items()})

    # log model artifact
    mlflow.sklearn.log_model(model, "model")

    # store the run_id so evaluate.py can log to the SAME run
    with open("models/mlflow_run_id.txt", "w") as f:
        f.write(mlflow.active_run().info.run_id)

# ── save model for DVC ────────────────────────────────────────
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print(f"Model saved to models/model.pkl")
