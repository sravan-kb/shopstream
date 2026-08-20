from airflow import DAG
from airflow.providers.databricks.operators.databricks import DatabricksRunNowOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "sravan",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="profit_prediction_batch",
    default_args=default_args,
    start_date=datetime(2026, 8, 1),
    schedule=timedelta(minutes=25),
    catchup=False,
    tags=["shopstream", "databricks", "profit"],
) as dag:

    trigger_profit_prediction = DatabricksRunNowOperator(
        task_id="trigger_profit_prediction",
        databricks_conn_id="databricks_backup2",
        job_id=250893043780169,
    )

    trigger_profit_prediction
