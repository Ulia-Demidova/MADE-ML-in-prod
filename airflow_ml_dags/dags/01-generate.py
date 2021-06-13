from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from .utils import default_args, VOLUME

with DAG(
    dag_id="01-generate",
    default_args=default_args,
    sheduler_interval="@daily",
    start_date=days_ago(1)
) as dag:
    start = DummyOperator(task_id="start_generating")
    generate = DockerOperator(
        image="airflow-generate",
        command="/data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-generate",
        do_xcom_push=False,
        volumes=[VOLUME]
    )
    start >> generate




