# T5CustomDataset Doc





# Usage

''' python
classification_dataset = T5DynamicDataset(
    tokenizer=tokenizer,
    data=classification_data,
    max_len=128,
    task_type='classification',
    input_text='sentence',
    target_text='label'
    # No need for task_specific_params in classification
)

'''
