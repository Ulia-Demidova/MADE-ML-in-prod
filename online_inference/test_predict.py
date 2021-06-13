import os
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.metrics import accuracy_score

from app import app

PATH_TO_DATASET = 'heart.csv'
TARGET_COLUMN = 'target'

os.environ['PATH_TO_MODEL'] = './model.pkl'


def test_predict():
    df = pd.read_csv(PATH_TO_DATASET)
    data = df.drop(columns=TARGET_COLUMN)
    target = df[TARGET_COLUMN]
    request_params = {
        'data': data.values.tolist(),
        'features': data.columns.tolist()
    }
    with TestClient(app) as client:
        response = client.get("/predict/", json=request_params)
        assert response.status_code == 200
        predictions = [x['disease'] for x in response.json()]
        accuracy = accuracy_score(target, predictions)
        assert 0.8 < accuracy < 1.0
