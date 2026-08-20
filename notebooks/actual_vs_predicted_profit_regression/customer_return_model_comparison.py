from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    Imputer,
    StandardScaler
)
from pyspark.ml.classification import RandomForestClassifier, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array

def main():
    spark = SparkSession.builder \
        .appName("ReturnPrediction_ModelComparison") \
        .enableHiveSupport() \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    source_table = "group_project.customer_ml_features"
    rf_model_save_path = "/group_project/models/customer_return_rf_pipeline_model"
    mlp_model_save_path = "/group_project/models/customer_return_mlp_pipeline_model"
    threshold_results_path = "/group_project/output/customer_model_threshold_results"
    final_predictions_path = "/group_project/output/customer_final_predictions_best_model"
    best_metrics_table = "group_project.customer_return_best_model_metrics"

    df = spark.table(source_table)
    df = df.toDF(*[c.lower() for c in df.columns])

    print("Source columns:")
    print(df.columns)

    if "ml_returned_line_count" not in df.columns:
        raise ValueError("ml_returned_line_count column not found in source table")

    df = df.withColumn(
        "returned_flag",
        F.when(F.col("ml_returned_line_count") > 0, 1).otherwise(0)
    )

    drop_cols = [
        "customer_id",
        "ml_returned_line_count",
        "total_returns",
        "last_order_date",
        "first_order_date"
    ]
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(*existing_drop_cols)

    df = df.withColumn("returned_flag", F.col("returned_flag").cast(DoubleType()))

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    label_col = "returned_flag"

    categorical_cols = [
        "membership_status",
        "age_bucket",
        "rfm_segment",
        "customer_segment",
        "favorite_product_category"
    ]
    categorical_cols = [c for c in categorical_cols if c in train_df.columns]

    numeric_cols = [
        c for c, t in train_df.dtypes
        if c != label_col and c not in categorical_cols and t in
        ("int", "bigint", "double", "float", "decimal", "long", "short")
    ]

    print("Numeric columns:", numeric_cols)
    print("Categorical columns:", categorical_cols)

    if len(categorical_cols) == 0 and len(numeric_cols) == 0:
        raise ValueError("No usable feature columns found after preprocessing")

    preprocess_stages = []

    if numeric_cols:
        imputer = Imputer(
            inputCols=numeric_cols,
            outputCols=[f"{c}_imputed" for c in numeric_cols]
        )
        preprocess_stages.append(imputer)
        numeric_feature_cols = [f"{c}_imputed" for c in numeric_cols]
    else:
        numeric_feature_cols = []

    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in categorical_cols
    ]

    encoders = [
        OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe")
        for c in categorical_cols
    ]

    preprocess_stages.extend(indexers)
    preprocess_stages.extend(encoders)

    assembler_inputs = numeric_feature_cols + [f"{c}_ohe" for c in categorical_cols]

    assembler = VectorAssembler(
        inputCols=assembler_inputs,
        outputCol="features_raw",
        handleInvalid="keep"
    )

    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withStd=True,
        withMean=False
    )

    preprocess_stages.append(assembler)
    preprocess_stages.append(scaler)

    rf = RandomForestClassifier(
        labelCol=label_col,
        featuresCol="features",
        predictionCol="prediction",
        probabilityCol="probability",
        rawPredictionCol="rawPrediction",
        numTrees=300,
        maxDepth=10,
        seed=42
    )

    rf_pipeline = Pipeline(stages=preprocess_stages + [rf])
    rf_model = rf_pipeline.fit(train_df)
    rf_model.write().overwrite().save(rf_model_save_path)
    print(f"Saved RandomForest model to: {rf_model_save_path}")

    rf_model_loaded = PipelineModel.load(rf_model_save_path)
    rf_pred = rf_model_loaded.transform(test_df)

    sample_row = rf_pred.select("features").first()
    input_dim = len(sample_row["features"])

    mlp = MultilayerPerceptronClassifier(
        labelCol=label_col,
        featuresCol="features",
        predictionCol="prediction",
        probabilityCol="probability",
        rawPredictionCol="rawPrediction",
        layers=[input_dim, 64, 32, 2],
        maxIter=100,
        blockSize=128,
        seed=42
    )

    mlp_pipeline = Pipeline(stages=preprocess_stages + [mlp])
    mlp_model = mlp_pipeline.fit(train_df)
    mlp_model.write().overwrite().save(mlp_model_save_path)
    print(f"Saved MLP model to: {mlp_model_save_path}")

    mlp_model_loaded = PipelineModel.load(mlp_model_save_path)
    mlp_pred = mlp_model_loaded.transform(test_df)

    def evaluate_thresholds(pred_df, model_name, thresholds):
        rows = []

        scored = pred_df.select(
            F.col(label_col).cast("int").alias("label"),
            vector_to_array(F.col("probability"))[1].alias("prob_1")
        )

        for th in thresholds:
            tmp = scored.withColumn(
                "pred_label",
                F.when(F.col("prob_1") >= F.lit(th), 1).otherwise(0)
            )

            cm = tmp.groupBy("label", "pred_label").count().collect()
            cm_dict = {(r["label"], r["pred_label"]): r["count"] for r in cm}

            tn = cm_dict.get((0, 0), 0)
            fp = cm_dict.get((0, 1), 0)
            fn = cm_dict.get((1, 0), 0)
            tp = cm_dict.get((1, 1), 0)

            total = tp + tn + fp + fn
            accuracy = (tp + tn) / total if total else 0.0

            precision_1 = tp / (tp + fp) if (tp + fp) else 0.0
            recall_1 = tp / (tp + fn) if (tp + fn) else 0.0
            f1_1 = (2 * precision_1 * recall_1 / (precision_1 + recall_1)) if (precision_1 + recall_1) else 0.0

            precision_0 = tn / (tn + fn) if (tn + fn) else 0.0
            recall_0 = tn / (tn + fp) if (tn + fp) else 0.0
            f1_0 = (2 * precision_0 * recall_0 / (precision_0 + recall_0)) if (precision_0 + recall_0) else 0.0

            macro_f1 = (f1_0 + f1_1) / 2.0
            balanced_accuracy = (recall_0 + recall_1) / 2.0
            recall_gap = abs(recall_0 - recall_1)

            rows.append((
                model_name, th, accuracy, balanced_accuracy, macro_f1,
                precision_0, recall_0, f1_0,
                precision_1, recall_1, f1_1,
                recall_gap, tn, fp, fn, tp
            ))

        return spark.createDataFrame(
            rows,
            schema=[
                "model", "threshold", "accuracy", "balanced_accuracy", "macro_f1",
                "precision_0", "recall_0", "f1_0",
                "precision_1", "recall_1", "f1_1",
                "recall_gap", "tn", "fp", "fn", "tp"
            ]
        )

    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]

    rf_threshold_results = evaluate_thresholds(rf_pred, "RandomForest", thresholds)
    mlp_threshold_results = evaluate_thresholds(mlp_pred, "MLP", thresholds)
    all_results = rf_threshold_results.unionByName(mlp_threshold_results)

    auc_evaluator = BinaryClassificationEvaluator(
        labelCol=label_col,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    rf_auc = auc_evaluator.evaluate(rf_pred)
    mlp_auc = auc_evaluator.evaluate(mlp_pred)

    print(f"RandomForest ROC AUC: {rf_auc:.4f}")
    print(f"MLP ROC AUC: {mlp_auc:.4f}")

    all_results.orderBy(
        F.desc("macro_f1"),
        F.desc("accuracy")
    ).show(50, truncate=False)

    best_row = all_results.orderBy(
        F.desc("macro_f1"),
        F.desc("accuracy")
    ).first()

    best_model_name = best_row["model"]
    best_threshold = best_row["threshold"]

    print("\n===== BEST MODEL SELECTION =====")
    print(f"Best Model        : {best_model_name}")
    print(f"Best Threshold    : {best_threshold}")
    print(f"Best Macro F1     : {best_row['macro_f1']:.4f}")
    print(f"Accuracy          : {best_row['accuracy']:.4f}")
    print(f"Balanced Accuracy : {best_row['balanced_accuracy']:.4f}")
    print(f"Recall Gap        : {best_row['recall_gap']:.4f}")
    print(f"TN={best_row['tn']} FP={best_row['fp']} FN={best_row['fn']} TP={best_row['tp']}")

    final_pred_df = rf_pred if best_model_name == "RandomForest" else mlp_pred
    best_model_path = rf_model_save_path if best_model_name == "RandomForest" else mlp_model_save_path
    best_auc = rf_auc if best_model_name == "RandomForest" else mlp_auc

    final_output = final_pred_df.withColumn(
        "return_probability",
        vector_to_array(F.col("probability"))[1]
    ).withColumn(
        "predicted_returned_best",
        F.when(F.col("return_probability") >= F.lit(best_threshold), 1).otherwise(0)
    )

    final_output.select(
        label_col,
        "return_probability",
        "predicted_returned_best"
    ).show(20, truncate=False)

    all_results.coalesce(1).write.mode("overwrite").option("header", True).csv(threshold_results_path)
    final_output = final_pred_df.withColumn(
    "return_probability",
    vector_to_array(F.col("probability"))[1]
    ).withColumn(
    "predicted_returned_best",
    F.when(F.col("return_probability") >= F.lit(best_threshold), 1).otherwise(0)
    )

    final_export = final_output.select(
    F.col(label_col).alias("returned_flag"),
    "return_probability",
    "predicted_returned_best"
    )

    final_export.coalesce(1).write.mode("overwrite").option("header", True).csv(final_predictions_path)

    best_metrics_data = [(
        source_table,
        best_model_name,
        best_model_path,
        float(best_threshold),
        float(best_auc),
        float(best_row["accuracy"]),
        float(best_row["balanced_accuracy"]),
        float(best_row["macro_f1"]),
        float(best_row["recall_gap"]),
        int(best_row["tn"]),
        int(best_row["fp"]),
        int(best_row["fn"]),
        int(best_row["tp"])
    )]

    best_metrics_df = spark.createDataFrame(
        best_metrics_data,
        [
            "source_table", "best_model_name", "best_model_path", "best_threshold",
            "roc_auc", "accuracy", "balanced_accuracy", "macro_f1", "recall_gap",
            "tn", "fp", "fn", "tp"
        ]
    )

    best_metrics_df.write.mode("overwrite").saveAsTable(best_metrics_table)

    print("\nSaved:")
    print(threshold_results_path)
    print(final_predictions_path)
    print(best_metrics_table)

    spark.stop()

if __name__ == "__main__":
    main()
