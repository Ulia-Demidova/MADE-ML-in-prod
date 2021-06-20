from datetime import timedelta

default_args = {
    "owner": "Yulia Demidova",
    "email": ["yuliaademidova@gmail.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

VOLUME = "/Users/yuliya/MADE/ML-in-prod/ulia-demidova/airflow_ml_dags/data:/data"
