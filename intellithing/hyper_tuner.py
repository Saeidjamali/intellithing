import optuna

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,  # for sequence classification tasks
    AutoModelForCausalLM,                # for causal language models (e.g., GPT-2)
    AutoModelForSeq2SeqLM,               # for sequence-to-sequence models (e.g., T5, BART)
    AutoModelForTokenClassification,     # for token classification tasks (e.g., NER, POS tagging)
    AutoTokenizer,
    DataCollatorForSeq2Seq,              # if you're working with seq2seq tasks
    Trainer,
    TrainingArguments,
    default_data_collator,               # default data collator handles batching, padding, etc.
)

from sklearn.model_selection import train_test_split  # for data splitting


class HyperparameterTuner:
    def __init__(self, model_name, tokenizer, train_dataset, val_dataset):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.best_params = None

        self.config = AutoConfig.from_pretrained(model_name)
        self.model_type = self.config.model_type
        
        
    def _load_model(self):
        """
        Load the appropriate model based on the model_type. Additional handling for different models
        can be added as needed.
        """
        config = AutoConfig.from_pretrained(self.model_name)
        model_type = config.model_type  # Identify the type of model

        # A dictionary to hold model classes for different types of tasks.
        # This dictionary can expand as we encounter new types of tasks.
        model_classes = {
            "sequence_classification": AutoModelForSequenceClassification,
            "causal_lm": AutoModelForCausalLM,
            "seq2seq_lm": AutoModelForSeq2SeqLM,
            "token_classification": AutoModelForTokenClassification,
            # Add other task types here...
        }

        # Here, we handle a variety of model types. This is not exhaustive and should be expanded as needed.
        if model_type in ["bert", "roberta", "distilbert", "electra", "xlnet", "transfo-xl", "reformer", "longformer", "deberta", "deberta-v2", "bigbird"]:
            # The majority are sequence classification tasks. Adjust as necessary for your use case.
            model_class = model_classes["sequence_classification"]

        elif model_type == "gpt2":
            model_class = model_classes["causal_lm"]

        elif model_type in ["t5", "bart", "pegasus", "bigbird_pegasus"]:
            model_class = model_classes["seq2seq_lm"]

        elif model_type == "layoutlm":
            model_class = model_classes["token_classification"]

        # ... other specific model type handling ...

        else:
            raise ValueError(f"Unsupported model type: {model_type}. Add appropriate handling.")

        # Load the model with the determined class.
        model = model_class.from_pretrained(self.model_name, config=config)
        return model


    def objective(self, trial):
        # Hyperparameter settings remain unchanged
        lr = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
        per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [2, 4, 8])
        warmup_steps = trial.suggest_int("warmup_steps", 0, 1000)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 0.3)
        num_train_epochs = trial.suggest_int("num_train_epochs", 1, 5)
        adam_epsilon = trial.suggest_float("adam_epsilon", 1e-8, 1e-6)
        gradient_accumulation_steps = trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4])



        # Setting training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=num_train_epochs,
            learning_rate=lr,
            per_device_train_batch_size=per_device_train_batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            adam_epsilon=adam_epsilon,
            gradient_accumulation_steps=gradient_accumulation_steps,
            evaluation_strategy="steps",
            eval_steps=100,
            logging_dir='./logs',
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
        )


        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
        )

        trainer.train()
        evaluation = trainer.evaluate()

        return evaluation['eval_loss']

    def tune_hyperparameters(self, n_trials=10, subset_size=0.1, timeout=None):
        if subset_size <= 0 or subset_size > 1:
            raise ValueError("subset_size should be greater than 0 and less than or equal to 1")

        # Subsetting the dataset for the tuning process
        train_subset, _ = train_test_split(self.train_dataset, train_size=subset_size, shuffle=True, stratify=None)

        # Adjusting the training dataset for the tuning process
        self.train_dataset = train_subset

        # Creating a study and optimizing the objective
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials, timeout=timeout)

        self.best_params = study.best_params
        return self.best_params


# Usage example:
# You need to ensure 'model_name', 'tokenizer', 'train_dataset', and 'val_dataset' are properly defined before this step.
# tuner = HyperparameterTuner(model_name, tokenizer, train_dataset, val_dataset)
# best_params = tuner.tune_hyperparameters(n_trials=10, subset_size=0.1)  # Adjust as necessary
