from setuptools import find_packages, setup


with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Homework1',
    author='Yuliya Demidova',
    install_requires=required,
    license='MIT',
)
