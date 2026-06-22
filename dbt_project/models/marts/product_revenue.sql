WITH orders AS (
    SELECT * FROM {{ ref('stg_orders') }}
)

SELECT
    stock_code,
    MAX(product_description)            AS product_description,
    SUM(quantity)                        AS total_units_sold,
    ROUND(SUM(line_revenue), 2)         AS total_revenue,
    COUNT(DISTINCT invoice_id)          AS times_ordered,
    COUNT(DISTINCT customer_id)         AS unique_customers,
    ROUND(AVG(unit_price), 2)           AS avg_unit_price

FROM orders
GROUP BY stock_code
ORDER BY total_revenue DESC