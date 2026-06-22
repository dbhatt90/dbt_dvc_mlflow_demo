# src/export_from_dbt.py
import duckdb
import os

os.makedirs("data/raw", exist_ok=True)

con = duckdb.connect("dbt_project/dev.duckdb")

# Export the mart dbt built
df = con.execute("SELECT * FROM customer_features").fetchdf()
df.to_csv("data/raw/customer_features.csv", index=False)

print(f"Exported {len(df)} rows to data/raw/customer_features.csv")
print(df.head())