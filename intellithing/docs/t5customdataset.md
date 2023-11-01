# T5CustomDataset Doc





# Usage


## Classification
``` python
classification_dataset = T5DynamicDataset(
    tokenizer=tokenizer,
    data=classification_data_df,
    max_len=128,
    task_type='classification',
    input_text='sentence',
    target_text='label'
    # No need for task_specific_params in classification
)

```

## Regression

``` python
regression_dataset = T5DynamicDataset(
    tokenizer=tokenizer,
    data=regression_data_df,
    max_len=128,
    task_type='regression',
    input_text='sentence',
    target_text='score'
    # No need for task_specific_params in regression
)
```

## Translation

``` python
translation_dataset = T5DynamicDataset(
    tokenizer=tokenizer,
    data=translation_data_df,
    max_len=512,
    task_type='translation',
    task_specific_params={'source_max_len': 256, 'target_max_len': 256},
    input_text='source_language_text',
    target_text='target_language_text'
)
```

## Summarization

``` python
summarization_dataset = T5DynamicDataset(
    tokenizer=tokenizer,
    data=summarization_data_df,
    max_len=512,
    task_type='summarization',
    task_specific_params={'source_max_len': 512, 'target_max_len': 150},
    input_text='document',
    target_text='summary'
)
```

## Questioning and Answering

``` python
qa_dataset = T5DynamicDataset(
    tokenizer=tokenizer,
    data=qa_data_df,
    max_len=512,
    task_type='question_answering',
    task_specific_params={'target_max_len': 128},  # Adjust target_max_len as needed
    input_text='question',
    target_text='answer'
)
```
## Text generation
``` python
text_generation_dataset = T5DynamicDataset(
    tokenizer=tokenizer,
    data=text_generation_data_df,
    max_len=256,
    task_type='text_generation',
    input_text='prompt',
    target_text='continuation'
    # No need for task_specific_params in text generation unless you have specific requirements
)
```

## Name Entity Recognition
``` python
ner_dataset = T5DynamicDataset(
    tokenizer=tokenizer,
    data=ner_data_df,
    max_len=128,
    task_type='ner',
    input_text='sentence',
    target_text='tags'
    # The align_labels_with_tokens function will be used here, no extra task_specific_params needed
)
```
## Part speech tagging
``` python
pos_dataset = T5DynamicDataset(
    tokenizer=tokenizer,
    data=pos_data_df,
    max_len=128,
    task_type='pos_tagging',
    input_text='sentence',
    target_text='tags'
    # The align_labels_with_tokens function will be used here, no extra task_specific_params needed
)
```
