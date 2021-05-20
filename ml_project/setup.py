from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Homework1',
    author='Yuliya Demidova',
    install_requires=[
        "click==7.1.2",
        "python-dotenv>=0.5.1",
        "scikit-learn==0.22.1",
        "dataclasses==0.6",
        "pyyaml==3.11",
        "marshmallow-dataclass==8.3.0",
        "pandas==1.2.3",
        "numpy==1.20.2",
        "pytest==6.0.1"
    ],
    license='MIT',
)
