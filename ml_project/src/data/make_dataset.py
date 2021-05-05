import pandas as pd
from sklearn.model_selection import train_test_split
from src.enities.splitting_params import SplittingParams
from typing import Tuple


def read_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def split_train_val(
        data: pd.DataFrame,
        params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(data, test_size=params.val_size, random_state=params.random_state)
