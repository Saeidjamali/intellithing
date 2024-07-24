from setuptools import setup, find_packages

setup(
    name="intellithing",
    version="0.11",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "matplotlib",
        "optuna",
        "scikit-learn",
        "torch",
        "fuzzywuzzy",
        "sentence-transformers",
        "fastapi",
        "uvicorn",
        "requests",
        "PyPDF2",
        "sqlalchemy"
    ],
)

