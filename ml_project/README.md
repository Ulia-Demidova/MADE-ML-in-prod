Homework 1
==============================

Installation:
~~~
python -m venv .venv
source .venv/bin/activate
pip install -e .
~~~
Train:
~~~
python train.py configs/train_logred_config.yaml
~~~
Predict:
~~~
python predict.py configs/predict_config.yaml
~~~
Test:
~~~
pytest tests/
~~~

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── configs            <- YAML configs for training and predictions
    │
    ├── logs               <- Logs storage
    │
    ├── data
    │   └── raw            <- The original, immutable data dump.
    ├── models             <- Trained and serialized models, model predictions
    │
    ├── notebooks          <- Jupyter notebooks. 
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │
    │   ├── enities        <- Scripts to create enities
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │
    │   └── models         <- Scripts to train models and then use trained models to make
    │                          predictions and evaluations
    └── tests              <- Directory with test scripts
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
