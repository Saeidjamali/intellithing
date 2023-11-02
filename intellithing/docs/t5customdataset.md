# T5CustomDataset Doc





# Usage


## Classification
``` python
classification_dataset = T5ClassificationDataset(
    tokenizer=tokenizer, 
    data=data_frame, 
    input_column='input_text', 
    target_column='class_label'
)

```

## Regression

``` python
regression_dataset = T5RegressionDataset(
    tokenizer=tokenizer, 
    data=data_frame, 
    input_column='input_text', 
    target_column='numerical_target'
)
```

## Translation

``` python
translation_dataset = T5TranslationDataset(
    tokenizer=tokenizer, 
    data_frame=data_frame, 
    source_text_column='source_language_text', 
    target_text_column='target_language_text'
)
```

## Summarization

``` python
summarization_dataset = T5SummarizationDataset(
    tokenizer=tokenizer, 
    data=data_frame, 
    text_column='input_text', 
    summary_column='target_summary'
)
```

## Questioning and Answering

``` python
qa_dataset = T5QADataset(
    tokenizer=tokenizer, 
    data=data_frame, 
    context_column='context', 
    question_column='question', 
    answer_column='answer'
)
```
## Text generation
``` python
text_gen_dataset = T5TextGenerationDataset(
    tokenizer=tokenizer, 
    data=data_frame, 
    text_column='input_text', 
    target_column='target_text'
)
```

## Name Entity Recognition
``` python
ner_dataset = T5NERDataset(
    tokenizer=tokenizer, 
    data=data_frame, 
    text_column='input_text', 
    ner_tags_column='ner_tags'
)
```

