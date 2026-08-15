-- Run these queries after creating the database with: python database.py

-- 1. Total demand and number of units sold for each product.
SELECT
    "Product ID" AS product_id,
    SUM("Demand") AS total_demand,
    SUM("Units Sold") AS total_units_sold
FROM sales_data
GROUP BY "Product ID"
ORDER BY total_demand DESC;

-- 2. Average demand by region.
SELECT
    "Region" AS region,
    ROUND(AVG("Demand"), 2) AS average_demand
FROM sales_data
GROUP BY "Region"
ORDER BY average_demand DESC;
