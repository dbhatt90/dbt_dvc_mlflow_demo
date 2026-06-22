WITH orders AS (
    SELECT * FROM {{ref('stg_orders')}}
),

max_date AS (
    SELECT MAX(invoice_date) AS snapshot_date 
    FROM orders
),

customer_agg AS (
    SELECT o.customer_id, m.snapshot_date,

    DATEDIFF('day', MAX(o.invoice_date), m.snapshot_date) AS recency_days,

    COUNT(DISTINCT o.invoice_id) AS frequency,

    ROUND(SUM(o.line_revenue),2) AS monetary_value,

    ROUND(SUM(o.line_revenue)/COUNT(DISTINCT o.invoice_id),2) AS avg_order_value,

    DATEDIFF('day', MIN(o.invoice_date),MAX(o.invoice_date)) AS tenure_days,

    COUNT(DISTINCT o.stock_code) AS unique_products,

    MODE(o.country) AS primary_country,

    FROM orders AS o
    CROSS JOIN max_date AS m 
    GROUP BY o.customer_id, m.snapshot_date
)

SELECT * FROM customer_agg