WITH orders AS (
    SELECT * FROM {{ ref('stg_orders') }}
)

SELECT
    DATE_TRUNC('month', invoice_date)   AS month,
    COUNT(DISTINCT invoice_id)          AS num_orders,
    COUNT(DISTINCT customer_id)         AS num_customers,
    SUM(quantity)                       AS total_units,
    ROUND(SUM(line_revenue), 2)         AS total_revenue,
    ROUND(AVG(line_revenue), 2)         AS avg_line_revenue

FROM orders
GROUP BY DATE_TRUNC('month', invoice_date)
ORDER BY month