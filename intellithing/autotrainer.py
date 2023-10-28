import os
import optuna
from intellithing.hyper_tuner import HyperparameterTuner
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split


class AutoTrainer(HyperparameterTuner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_model = None

    def auto_train(self, n_trials=10, subset_size=0.1, timeout=None):
        # Tune hyperparameters on a subset of the training data
        best_params = self.tune_hyperparameters(n_trials=n_trials, subset_size=subset_size, timeout=timeout)

        # Load the model again, as it was trained on the subset of data during hyperparameter tuning
        self.model = self._load_model()

        # Set the best_params from tuning to the TrainingArguments
        training_args = TrainingArguments(
            output_dir='./results_full',
            num_train_epochs=best_params['num_train_epochs'],
            learning_rate=best_params['lr'],
            per_device_train_batch_size=best_params['per_device_train_batch_size'],
            warmup_steps=best_params['warmup_steps'],
            weight_decay=best_params['weight_decay'],
            adam_epsilon=best_params['adam_epsilon'],
            gradient_accumulation_steps=best_params['gradient_accumulation_steps'],
            evaluation_strategy="epoch",
            logging_dir='./logs_full',
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
        )

        # Initialize the Trainer with the appropriate parameters
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,  # Using the full training set
            eval_dataset=self.val_dataset,
        )

        # Train the model on the full dataset
        trainer.train()

        # Save the best model (which is loaded automatically by Trainer if 'load_best_model_at_end' is True)
        self.best_model = self.model

        # Save the model to a directory
        save_directory = "finetuned_model"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        self.best_model.save_pretrained(save_directory)

        return self.best_model

# Usage example:
# Assuming 'model_name', 'tokenizer', 'train_dataset', and 'val_dataset' are defined
# auto_trainer = AutoTrainer(model_name, tokenizer, train_dataset, val_dataset)
# best_model = auto_trainer.auto_train(n_trials=10, subset_size=0.1)  # Adjust n_trials and subset_size as necessary

