from setuptools import setup, find_packages

setup(
    name="intellithing",
    version="0.10",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "matplotlib",
        "optuna",
        "scikit-learn",
        "torch"
    ],
)

