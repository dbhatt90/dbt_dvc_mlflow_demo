WITH source AS (
    SELECT * FROM {{ref('raw_orders')}}
),

cleaned AS (
    SELECT
    CAST(Invoice AS VARCHAR)        AS invoice_id,
    CAST(StockCode AS VARCHAR)      AS stock_code,
    Description                     AS product_description,
    Quantity                        AS quantity,
    CAST(InvoiceDate AS TIMESTAMP)  AS invoice_date,
    Price                           AS unit_price,
    CAST("Customer ID" AS VARCHAR)  AS customer_id,
    Country                         AS country,
    ROUND(Quantity*Price,2)         AS line_revenue

    FROM source

    WHERE CAST(Invoice AS VARCHAR) NOT LIKE 'C%'
    AND Quantity > 0 AND Price>0 AND "Customer ID" IS NOT NULL
)

SELECT * FROM cleaned