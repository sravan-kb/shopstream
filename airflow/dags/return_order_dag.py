from airflow import DAG
from airflow.providers.databricks.operators.databricks import DatabricksRunNowOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "sravan",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="return_order_batch_prediction_backup2",
    default_args=default_args,
    start_date=datetime(2026, 8, 1),
    schedule=timedelta(hours=8),
    catchup=False,
    tags=["shopstream", "databricks", "backup2"],
) as dag:

    run_return_prediction = DatabricksRunNowOperator(
        task_id="trigger_return_order_prediction",
        databricks_conn_id="databricks_backup2",
        job_id=899711633151099,
    )

    run_return_prediction
