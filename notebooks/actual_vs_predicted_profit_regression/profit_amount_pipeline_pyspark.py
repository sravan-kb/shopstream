import os
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PySpark Imports
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.regression import (
    LinearRegression,
    GeneralizedLinearRegression,
    DecisionTreeRegressor,
    RandomForestRegressor,
    FMRegressor,
)
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

# -------------------------------------------------------------------------
# 1. Initialize Spark Session
# -------------------------------------------------------------------------
spark = SparkSession.builder \
    .appName("Ecommerce_Orders_ML_Pipeline") \
    .getOrCreate()

# Load Dataset
df = spark.read.csv('ecommerce_orders_dataset.csv', header=True, inferSchema=True)

# -------------------------------------------------------------------------
# 2. Categorical Encoding & Feature Engineering Pipeline
# -------------------------------------------------------------------------
categorical_cols = [
    'Customer_Gender', 'Country', 'City', 'Customer_Segment', 
    'Product_Category', 'Product_Subcategory', 'Brand', 'Coupon_Used', 
    'Payment_Method', 'Device_Type', 'Traffic_Source', 'Membership_Status', 
    'Shipping_Method', 'Warehouse_Region', 'Order_Status', 'Returned', 
    'Season', 'Holiday_Season', 'High_Value_Order'
]

features = [
    'Customer_Age', 'Customer_Gender_idx', 'Country_idx', 'Customer_Segment_idx', 
    'Product_Category_idx', 'Brand_idx', 'Unit_Price', 'Quantity', 'Discount_Percent', 
    'Shipping_Cost', 'Payment_Method_idx', 'Device_Type_idx', 'Traffic_Source_idx', 
    'Membership_Status_idx', 'Shipping_Method_idx', 'Warehouse_Region_idx', 
    'Season_idx', 'Holiday_Season_idx'
]

# Create StringIndexers for categorical features (PySpark equivalent to LabelEncoder)
indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") 
    for c in categorical_cols
]

# Assemble features into a single Vector column
assembler = VectorAssembler(inputCols=features, outputCol="unscaled_features")

# Feature Scaling
scaler = StandardScaler(inputCol="unscaled_features", outputCol="features", withStd=True, withMean=True)

# Fit Pipeline for Data Preprocessing
preprocessing_pipeline = Pipeline(stages=indexers + [assembler, scaler])
pipeline_model = preprocessing_pipeline.fit(df)
df_transformed = pipeline_model.transform(df)

# Train / Test Split
train_df, test_df = df_transformed.randomSplit([0.8, 0.2], seed=42)
train_df.cache()
test_df.cache()

# -------------------------------------------------------------------------
# 3. Model Definition & Evaluation Loop
# -------------------------------------------------------------------------
models = {
    'Linear Regression': LinearRegression(featuresCol='features', labelCol='Profit_Amount'),
    'Ridge (L2 Regularized)': LinearRegression(featuresCol='features', labelCol='Profit_Amount', regParam=10.0, elasticNetParam=0.0),
    'Lasso (L1 Regularized)': LinearRegression(featuresCol='features', labelCol='Profit_Amount', regParam=0.5, elasticNetParam=1.0),
    'Elastic Net': LinearRegression(featuresCol='features', labelCol='Profit_Amount', regParam=0.5, elasticNetParam=0.5),
    'Decision Tree Regressor': DecisionTreeRegressor(featuresCol='unscaled_features', labelCol='Profit_Amount', maxDepth=10, seed=42),
    'Random Forest': RandomForestRegressor(featuresCol='unscaled_features', labelCol='Profit_Amount', numTrees=100, seed=42),
    'Factorization Machines (MLP Alt)': FMRegressor(featuresCol='features', labelCol='Profit_Amount', stepSize=0.01)
}

evaluator_r2 = RegressionEvaluator(labelCol="Profit_Amount", predictionCol="prediction", metricName="r2")
evaluator_mae = RegressionEvaluator(labelCol="Profit_Amount", predictionCol="prediction", metricName="mae")
evaluator_rmse = RegressionEvaluator(labelCol="Profit_Amount", predictionCol="prediction", metricName="rmse")

results_data = []

for name, model in models.items():
    fitted_model = model.fit(train_df)
    predictions = fitted_model.transform(test_df)
    
    r2 = evaluator_r2.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)
    rmse = evaluator_rmse.evaluate(predictions)
    
    results_data.append({"Model": name, "R2 Score": f"{r2:.4f}", "MAE": f"{mae:.2f}", "RMSE": f"{rmse:.2f}"})

# -------------------------------------------------------------------------
# 4. Fit Primary Model (Random Forest) for Diagnostic Plots
# -------------------------------------------------------------------------
rf_model = RandomForestRegressor(featuresCol='unscaled_features', labelCol='Profit_Amount', numTrees=100, seed=42)
rf_fitted = rf_model.fit(train_df)
rf_predictions = rf_fitted.transform(test_df)

# Convert evaluation sample to Pandas for visualization
plot_pd = rf_predictions.select("Profit_Amount", "prediction").toPandas()
y_test = plot_pd["Profit_Amount"]
rf_preds = plot_pd["prediction"]
residuals = y_test - rf_preds

# Plot Generation Helper (Encodes figures to Base64 HTML strings)
def fig_to_base64(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"

# Figure 1: Actual vs Predicted
fig1, ax1 = plt.subplots(figsize=(7, 5))
ax1.scatter(y_test, rf_preds, alpha=0.5, color='#1f77b4')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax1.set_xlabel("Actual Values")
ax1.set_ylabel("Predicted Values")
ax1.set_title("Random Forest - Actual vs Predicted")
img1 = fig_to_base64(fig1)

# Figure 2: Residual Plot
fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.scatter(rf_preds, residuals, alpha=0.5, color='#ff7f0e')
ax2.axhline(y=0, color='r', linestyle='--')
ax2.set_xlabel("Predicted Values")
ax2.set_ylabel("Residuals")
ax2.set_title("Random Forest - Residual Plot")
img2 = fig_to_base64(fig2)

# Figure 3: Feature Importance
importances = rf_fitted.featureImportances.toArray()
feat_imp = pd.Series(importances, index=features).sort_values(ascending=True)

fig3, ax3 = plt.subplots(figsize=(8, 6))
feat_imp.plot(kind='barh', ax=ax3, color='#2ca02c')
ax3.set_xlabel("Importance")
ax3.set_ylabel("Features")
ax3.set_title("Random Forest - Feature Importance")
img3 = fig_to_base64(fig3)

# Figure 4: Prediction Error Distribution
fig4, ax4 = plt.subplots(figsize=(7, 5))
ax4.hist(residuals, bins=30, color='#d62728', edgecolor='black', alpha=0.7)
ax4.axvline(x=0, color='k', linestyle='--')
ax4.set_xlabel("Prediction Error")
ax4.set_ylabel("Frequency")
ax4.set_title("Random Forest - Error Distribution")
img4 = fig_to_base64(fig4)

# -------------------------------------------------------------------------
# 5. Build and Serialize Webpage Output (HTML)
# -------------------------------------------------------------------------
results_df = pd.DataFrame(results_data)
table_html = results_df.to_html(classes='metrics-table', index=False)

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PySpark ML Model Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 30px; background-color: #f8f9fa; color: #333; }}
        h1, h2 {{ color: #2c3e50; }}
        .metrics-table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; background: white; }}
        .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        .metrics-table th {{ background-color: #34495e; color: white; }}
        .metrics-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .grid-container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .card {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .card img {{ width: 100%; height: auto; }}
    </style>
</head>
<body>
    <h1>PySpark Machine Learning Evaluation Output</h1>
    <h2>Model Performance Comparison</h2>
    {table_html}

    <h2>Random Forest Model Diagnostics</h2>
    <div class="grid-container">
        <div class="card"><img src="{img1}" alt="Actual vs Predicted"/></div>
        <div class="card"><img src="{img2}" alt="Residual Plot"/></div>
        <div class="card"><img src="{img3}" alt="Feature Importance"/></div>
        <div class="card"><img src="{img4}" alt="Error Distribution"/></div>
    </div>
</body>
</html>
"""

# Export HTML to file
output_path = "ml_evaluation_report.html"
with open(output_path, "w") as f:
    f.write(html_content)

print(f"Report generated successfully: {os.path.abspath(output_path)}")

# Stop Spark Session
spark.stop()