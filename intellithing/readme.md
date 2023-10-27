# LayerFreezer Documentation

`LayerFreezer` is a utility class designed to make freezing and unfreezing layers in various Hugging Face models more intuitive and streamlined.

## Table of Contents
1. [Supported Model Types](#supported-model-types)
2. [Usage](#usage)
   - [Initialization](#initialization)
   - [Method](#Methods)
   - [Examples](#Examples)
   
3. [Notes](#notes)

## Supported Model Types

- **T5 and BART**
  - Structure: Both encoder and decoder.
  - Naming convention: `encoder.block.{index}` and `decoder.block.{index}`
- **BERT, RoBERTa, DistilBERT, and Electra**
  - Structure: Encoder only.
  - Naming convention: `encoder.layer.{index}`
- **XLNet**
  - Structure: Encoder only.
  - Naming convention: `transformer.layer.{index}`
- **GPT-2 and TransformerXL**
  - Structure: No differentiation between encoder/decoder.
  - Naming convention: `h.{index}`



## Usage

### Initialization

To initialize the `LayerFreezer` class:

```python
from intellithing.layer_freezer import LayerFreezer
freezer = LayerFreezer(model_instance, model_type='model_name_here')
```

- `model_instance`: Your HuggingFace model instance.
- `model_type`: The type of the model you're using (e.g., 't5', 'bert', 'xlnet', etc.).

### Methods

#### Freeze/Unfreeze Specific Layers

To freeze or unfreeze specific layers in the model:

```python
# Freeze specific layers
layers_to_freeze = [0, 1, 2]
freezer.freeze_layers(layers_to_freeze, part='encoder')

# Unfreeze specific layers
layers_to_unfreeze = [10, 11]
freezer.unfreeze_layers(layers_to_unfreeze, part='decoder')
```

Parameters:
- `layer_indices`: A list of indices for the layers you want to freeze or unfreeze.
- `part`: The part of the model you're referring to - this can be 'encoder', 'decoder', or 'all'. By default, it's set to 'encoder'.

#### Freeze and Unfreeze All Layers

To freeze or unfreeze all layers in the model:

```python
# Freeze all layers
freezer.freeze_all()

# Unfreeze all layers
freezer.unfreeze_all()
```

## Examples

Here are some practical examples:

1. **Freeze all layers**:
```python
freezer.freeze_all()
```

2. **Unfreeze all layers**:
```python
freezer.unfreeze_all()
```

3. **Freeze specific layers in the encoder**:
```python
layers_to_freeze = [0, 1, 2]
freezer.freeze_layers(layers_to_freeze, part='encoder')
```

4. **Unfreeze specific layers in the decoder**:
```python
layers_to_unfreeze = [10, 11]
freezer.unfreeze_layers(layers_to_unfreeze, part='decoder')
```

## Note:
Please ensure the right model names are included. For example instead of "t5-base" write it as "t5"


-----------------

#Learning Rate finder

# LearningRateFinderCallback

## Table of Contents
1. [Summary](#summary)
2. [Usage](#usage)
    - [Importing the Callback](#importing-the-callback)
    - [Creating Datasets](#creating-datasets)
    - [Configuring Training Arguments](#configuring-training-arguments)
    - [Using the LearningRateFinderCallback](#using-the-learningratefindercallback)
    - [Visualizing Results](#visualizing-results)

## Summary
The `LearningRateFinderCallback` is a callback provided by the `intellithing` library that helps you find the optimal learning rate for your machine learning model during the training process. It performs a short training run with a range of learning rates and records the corresponding loss values. You can then visualize the results to determine an appropriate learning rate for your task.

## Usage

```python
# Import the LearningRateFinderCallback from intellithing
from intellithing.lr_finder import LearningRateFinderCallback

# Create your training and validation datasets
train_dataset = T5CustomDataset(tokenizer, train_df, 'input_text', 'target_text', source_max_len=512, target_max_len=256)
val_dataset = T5CustomDataset(tokenizer, val_df, 'input_text', 'target_text', source_max_len=512, target_max_len=256)

# Create a LearningRateFinderCallback with a specified learning rate range
lr_finder = LearningRateFinderCallback(start_lr=1e-7, end_lr=1)

# Configure your training arguments, including max_steps and batch sizes
training_args = TrainingArguments(
    output_dir='./t5base',
    max_steps=100,  # Ensure to use a small number of steps
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    report_to="none",  # Disable logging to Wandb or other platforms
    load_best_model_at_end=False,  # set this to False
)

# Create a Trainer with your model, datasets, and the LearningRateFinderCallback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[lr_finder],  # Add the LearningRateFinderCallback here
)

# Start the learning rate finder by running the trainer
trainer.train()

# After completion, you can check the generated plot to determine an appropriate learning rate.

***




# LayerFreezer Documentation

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
