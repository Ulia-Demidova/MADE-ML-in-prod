from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from .utils import default_args, VOLUME


with DAG(
    dag_id="03-predict",
    default_args=default_args,
    sheduler_interval="@daily",
    start_date=days_ago(1)
) as dag:
    predict = DockerOperator(
        image="airflow-predict",
        command="--input-dir /data/raw/{{ ds }} --model-dir {{ var.value.MODEL_DIR }} "
                "--output-dir /data/predictions/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        volumes=[VOLUME]
    )
