from setuptools import setup, find_packages

setup(
    name="intellithing",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "matplotlib"
        "torch"
    ],
)

