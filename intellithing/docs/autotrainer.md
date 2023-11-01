# Autotrainer

with intellithing autotrainer you can now find the best hyperparameters and automatically the start the finetuning with the same hyperparameters. 
 Supported Model Types

- **T5 and BART**

- **BERT, RoBERTa, DistilBERT, and Electra**

- **XLNet**

- **GPT-2 and TransformerXL**


#usage

```python
# Assuming 'model_name', 'tokenizer', 'train_dataset', and 'val_dataset' are defined
 auto_trainer = AutoTrainer(model_name, tokenizer, train_dataset, val_dataset)
 best_model = auto_trainer.auto_train(n_trials=10, subset_size=0.1)  # Adjust n_trials and subset_size as necessary
```

