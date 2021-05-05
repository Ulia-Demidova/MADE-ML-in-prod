from pandas.testing import assert_frame_equal
from src.data.make_dataset import read_data, split_train_val
from src.enities.splitting_params import SplittingParams


def test_read_data(tmpdir, sample_data):
    data_path = tmpdir.join('sample.csv')
    sample_data.to_csv(data_path, index_label=False)
    data = read_data(data_path)
    assert_frame_equal(sample_data, data)


def test_split_data(sample_data):
    val_size = 0.2
    split_params = SplittingParams(val_size=val_size, random_state=123)
    train, val = split_train_val(sample_data, split_params)
    assert len(train) > len(val)
