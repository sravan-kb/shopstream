

-- Retrieve prediction data for the dashboard
SELECT *
FROM workspace.default.ecommerce_predictions


-- Retrieve orders predicted as high-value
SELECT *
FROM workspace.default.ecommerce_predictions
WHERE prediction = 'Yes'


-- Count high-value orders by customer segment
SELECT
    Customer_Segment,
    COUNT(*) AS high_value_count
FROM workspace.default.ecommerce_predictions
WHERE prediction = 'Yes'
GROUP BY Customer_Segment
ORDER BY high_value_count DESC


-- Calculate total predicted profit
SELECT
ROUND(SUM(predicted_profit_amount),2) AS Total_Predicted_Profit
FROM workspace.default.ecommerce_profit_prediction;


-- Compare actual and predicted profit by month
SELECT
CASE Month
    WHEN 1 THEN 'Jan'
    WHEN 2 THEN 'Feb'
    WHEN 3 THEN 'Mar'
    WHEN 4 THEN 'Apr'
    WHEN 5 THEN 'May'
    WHEN 6 THEN 'Jun'
    WHEN 7 THEN 'Jul'
    WHEN 8 THEN 'Aug'
    WHEN 9 THEN 'Sep'
    WHEN 10 THEN 'Oct'
    WHEN 11 THEN 'Nov'
    WHEN 12 THEN 'Dec'
END AS Month_Name,
Month,
ROUND(SUM(Profit_Amount),2) AS Actual_Profit,
ROUND(SUM(predicted_profit_amount),2) AS Predicted_Profit
FROM workspace.default.ecommerce_profit_prediction
GROUP BY Month
ORDER BY Month;



