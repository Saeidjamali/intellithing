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


