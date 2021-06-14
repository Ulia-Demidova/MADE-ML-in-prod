from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from .utils import default_args, VOLUME


with DAG(
    dag_id="02-train-val",
    default_args=default_args,
    sheduler_interval="@weekly",
    start_date=days_ago(7)
) as dag:
    preprocess = DockerOperator(
        image="airflow-preprocess",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/processed/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        volumes=[VOLUME]
    )

    split = DockerOperator(
        image="airflow-split",
        command="--input-dir /data/processed/{{ ds }} --output-dir /data/splitted/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-split",
        do_xcom_push=False,
        volumes=[VOLUME]
    )

    train = DockerOperator(
        image="airflow-train",
        command="--data-dir /data/splitted/{{ ds }} --model-dir /data/models/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-train",
        do_xcom_push=False,
        volumes=[VOLUME]
    )

    val = DockerOperator(
        image="airflow-val",
        command="--data-dir /data/splitted/{{ ds }} --model-dir /data/models/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-val",
        do_xcom_push=False,
        volumes=[VOLUME]
    )

    preprocess >> split >> train >> val




