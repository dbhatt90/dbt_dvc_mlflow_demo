import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yaml
import os
import json

# load params
with open("params.yaml") as f:
    params = yaml.safe_load(f)["preprocess"]

TEST_SIZE = params["test_size"]
RANDOM_STATE = params["random_state"]
CHURN_DAYS = params["churn_days"]

# load dataset
df = pd.read_csv("data/raw/customer_features.csv")
print(f"Loaded {len(df)} customers data")
print(df.dtypes)

# engineer labels/target
# recency_days was computed in dbt as (snapshot_date - last_purchase_date)
# customers who haven't bought in CHURN_DAYS are labelled churned = 1

df["churned"] = (df['recency_days']>=CHURN_DAYS).astype(int)
print('\nChurn distribution:')
print(df['churned'].value_counts())
print(f"Churn rate: {df['churned'].mean():.2%}")

# feature selection
FEATURES = df.columns.drop(['snapshot_date', 'churned','primary_country']).tolist()

# country encoding
TOP_N_COUNTRIES = 5
top_countries = df['primary_country'].value_counts().nlargest(TOP_N_COUNTRIES).index 
df['primary_country_clean'] = df['primary_country'].where(
    df['primary_country'].isin(top_countries), other='other'
)
country_dummies = pd.get_dummies(df['primary_country_clean'],prefix='country',drop_first=True)
df = pd.concat([df,country_dummies],axis=1)
FEATURES+=list(country_dummies.columns)

print(f"\nFeatures used: {FEATURES}")

X = df[FEATURES].copy()
y = df["churned"].copy()

# ── handle any remaining nulls ────────────────────────────────
# stg_orders removed nulls but defensive check here
null_counts = X.isnull().sum()
if null_counts.any():
    print(f"\nNulls found, filling with median:\n{null_counts[null_counts > 0]}")
    X = X.fillna(X.median())

# ── train / test split ────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y        # preserve churn ratio in both splits
)

print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# ── scale numeric features ────────────────────────────────────
# country dummy columns are already 0/1, only scale continuous features
NUMERIC_FEATURES = [
    "recency_days",
    "frequency",
    "monetary_value",
    "avg_order_value",
    "tenure_days",
    "unique_products",
]

scaler = StandardScaler()
X_train[NUMERIC_FEATURES] = scaler.fit_transform(X_train[NUMERIC_FEATURES])
X_test[NUMERIC_FEATURES]  = scaler.transform(X_test[NUMERIC_FEATURES])


# ── save outputs ──────────────────────────────────────────────
os.makedirs("data/processed", exist_ok=True)

X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv",   index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv",   index=False)

# save scaler params so evaluate.py can use them if needed
scaler_params = {
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
    "features": NUMERIC_FEATURES,
}
with open("data/processed/scaler_params.json", "w") as f:
    json.dump(scaler_params, f, indent=2)

print("\nPreprocessing complete. Files written to data/processed/")