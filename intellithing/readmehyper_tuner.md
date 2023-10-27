# HyperParameterTuning

This project provides a tool for hyperparameter tuning of models available in the Hugging Face transformers library. It is designed to dynamically handle various types of transformer models and is particularly useful for tasks such as text classification, text generation, etc.

## Features

- Supports a wide range of transformer models from the Hugging Face library.
- Dynamic loading of models based on the specified task.
- Utilizes Optuna for efficient hyperparameter tuning.
- Easy integration and usage.

## Usage Instructions

1. **Data Preparation**:
Load your training and validation datasets.
```python
train_dataset, val_dataset = ...  # Insert your data loading mechanism here
```

2. **Initialize the Tuner**:
Set up the tuner with your model and tokenizer.
```python

from intellithing.hyper_tuner import HyperparameterTuner  

model_name = 'your_preferred_model'  # For example, 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

tuner = HyperparameterTuner(model_name, tokenizer, train_dataset, val_dataset)
```

3. **Run Hyperparameter Tuning**:
Invoke the tuning function to optimize the model's hyperparameters.
```python
best_parameters = tuner.tune_hyperparameters(n_trials=10, subset_size=0.1)  # Adjust 'n_trials' and data subset as needed
print("Optimal hyperparameters:", best_parameters)
```

