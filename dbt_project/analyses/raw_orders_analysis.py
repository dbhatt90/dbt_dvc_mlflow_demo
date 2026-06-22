import duckdb
import os
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "dev.duckdb")
con = duckdb.connect(DB_PATH)

# See the raw table
print(con.execute("SELECT * FROM stg_orders LIMIT 5").fetchdf())

# Check column names and types
print(con.execute("DESCRIBE raw_orders").fetchdf())

# How many rows total
print(con.execute("SELECT COUNT(*) FROM raw_orders").fetchdf())

# How many nulls in Customer ID
print(con.execute("""
    SELECT
        COUNT(*) AS total_rows,
        COUNT("Customer ID") AS non_null_customers,
        COUNT(*) - COUNT("Customer ID") AS null_customers
    FROM raw_orders
""").fetchdf())

# Check cancellations — invoices starting with C
print(con.execute("""
    SELECT
        SUM(CASE WHEN CAST(Invoice AS VARCHAR) LIKE 'C%' THEN 1 ELSE 0 END) AS cancellations,
        SUM(CASE WHEN CAST(Invoice AS VARCHAR) NOT LIKE 'C%' THEN 1 ELSE 0 END) AS valid_orders
    FROM raw_orders
""").fetchdf())

# Check negative quantities
print(con.execute("""
    SELECT COUNT(*) FROM raw_orders WHERE Quantity < 0
""").fetchdf())