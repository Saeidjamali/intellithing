from torch.utils.data import Dataset
import pandas as pd
import torch

class T5RegressionDataset(Dataset):
    def __init__(self, tokenizer, data, input_column, target_column, source_max_len=512, target_max_len=32, label_default_value=0.0):
        self.tokenizer = tokenizer
        self.data_frame = data
        self.input_column = input_column
        self.target_column = target_column
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
        self.label_default_value = label_default_value
        

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        try:
            # Attempt to handle text that is not in string format
            text = self.data_frame.iloc[index][self.input_column]
            if pd.isna(text):
                text = "Missing data"
            else:
                text = str(text)  # Convert any type to string

            # Handle missing or non-numeric label data dynamically
            label = self.data_frame.iloc[index][self.target_column]
            if pd.isna(label) or not self._is_float(label):
                label = self.label_default_value
            else:
                label = float(label)

            # Tokenize text
            tokenized_input = self.tokenizer(
                text,
                max_length=self.source_max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # Prepare inputs and labels
            input_ids = tokenized_input['input_ids'].squeeze().to(dtype=torch.long)  # Explicitly convert to torch.long here
            attention_mask = tokenized_input['attention_mask'].squeeze().to(dtype=torch.long)  # Same for attention_mask

            return {
                'input_ids': input_ids, 
                'attention_mask': attention_mask, 
                'labels': torch.tensor(label, dtype=torch.float)  # Use torch.float for regression
            }

        except Exception as e:
            raise RuntimeError(f"Error processing data at index {index}: {e}")

    @staticmethod
    def _is_float(value):
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

        from torch.utils.data import Dataset


class T5ClassificationDataset(Dataset):
    def __init__(self, tokenizer, data, input_column, target_column, source_max_len=512, target_max_len=32):
        self.tokenizer = tokenizer
        self.data_frame = data
        self.input_column = input_column
        self.target_column = target_column
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
        
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        # Extract the text and label from the DataFrame
        text = self.data_frame.iloc[index][self.input_column]
        text = str(text) if not pd.isna(text) else "Missing data"
        
        label = self.data_frame.iloc[index][self.target_column]
        label = str(label) if not pd.isna(label) else "Missing label"
        
        # Tokenize the text
        tokenized_input = self.tokenizer(
            text,
            max_length=self.source_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize the label
        tokenized_label = self.tokenizer(
            label,
            max_length=self.target_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Extract the input_ids and attention_mask and squeeze to remove the batch dimension
        input_ids = tokenized_input['input_ids'].squeeze().to(dtype=torch.long)
        attention_mask = tokenized_input['attention_mask'].squeeze().to(dtype=torch.long)
        
        # Extract the label_ids and squeeze to remove the batch dimension
        label_ids = tokenized_label['input_ids'].squeeze().to(dtype=torch.long)
        
        # Return a dictionary of tensors
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label_ids
        }



class T5QADataset(Dataset):
    def __init__(self, tokenizer, data, question_column, context_column, answer_column, source_max_len=512, target_max_len=32):
        self.tokenizer = tokenizer
        self.data_frame = data
        self.question_column = question_column
        self.context_column = context_column
        self.answer_column = answer_column
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
        
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        # Extract the question, context, and answer from the DataFrame
        question = self.data_frame.iloc[index][self.question_column]
        question = str(question) if not pd.isna(question) else "Missing question"
        
        context = self.data_frame.iloc[index][self.context_column]
        context = str(context) if not pd.isna(context) else "Missing context"
        
        answer = self.data_frame.iloc[index][self.answer_column]
        answer = str(answer) if not pd.isna(answer) else "Missing answer"
        
        # Format the input text as question followed by context
        t5_input = f"question: {question} context: {context}"
        
        # Tokenize the input text
        tokenized_input = self.tokenizer(
            t5_input,
            max_length=self.source_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize the answer
        tokenized_answer = self.tokenizer(
            answer,
            max_length=self.target_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Extract the input_ids and attention_mask and squeeze to remove the batch dimension
        input_ids = tokenized_input['input_ids'].squeeze().to(dtype=torch.long)
        attention_mask = tokenized_input['attention_mask'].squeeze().to(dtype=torch.long)
        
        # Extract the answer_ids and squeeze to remove the batch dimension
        answer_ids = tokenized_answer['input_ids'].squeeze().to(dtype=torch.long)
        
        # Return a dictionary of tensors
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': answer_ids
        }



class T5TextGenerationDataset(Dataset):
    def __init__(self, tokenizer, data, input_column, target_column, source_max_len=512, target_max_len=128):
        self.tokenizer = tokenizer
        self.data_frame = data
        self.input_column = input_column
        self.target_column = target_column
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
        
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        # Retrieve the text and the target from the dataframe
        text = str(self.data_frame.iloc[index][self.input_column]) if not pd.isna(self.data_frame.iloc[index][self.input_column]) else "Missing text"
        target = str(self.data_frame.iloc[index][self.target_column]) if not pd.isna(self.data_frame.iloc[index][self.target_column]) else "Missing target"

        # Tokenize the text and target
        tokenized_input = self.tokenizer(
            text,
            max_length=self.source_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        tokenized_target = self.tokenizer(
            target,
            max_length=self.target_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = tokenized_input['input_ids'].squeeze().to(dtype=torch.long)
        attention_mask = tokenized_input['attention_mask'].squeeze().to(dtype=torch.long)
        target_ids = tokenized_target['input_ids'].squeeze().to(dtype=torch.long)

        # Replace the padding token id's of the target by -100 so that it is ignored by the loss function
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': target_ids
        }



class T5SummarizationDataset(Dataset):
    def __init__(self, tokenizer, data, input_column, target_column, source_max_len=512, target_max_len=150):
        self.tokenizer = tokenizer
        self.data_frame = data
        self.input_column = input_column
        self.target_column = target_column
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
        
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        # Retrieve the text and the summary from the dataframe
        text = str(self.data_frame.iloc[index][self.input_column]) if not pd.isna(self.data_frame.iloc[index][self.input_column]) else "Missing text"
        summary = str(self.data_frame.iloc[index][self.target_column]) if not pd.isna(self.data_frame.iloc[index][self.target_column]) else "Missing summary"

        # Prefix the input text with "summarize: " as per T5's expected format
        t5_input = f"summarize: {text}"

        # Tokenize the text and summary
        tokenized_input = self.tokenizer(
            t5_input,
            max_length=self.source_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        tokenized_target = self.tokenizer(
            summary,
            max_length=self.target_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = tokenized_input['input_ids'].squeeze().to(dtype=torch.long)
        attention_mask = tokenized_input['attention_mask'].squeeze().to(dtype=torch.long)
        target_ids = tokenized_target['input_ids'].squeeze().to(dtype=torch.long)

        # Replace the padding token id's of the target by -100 so that it is ignored by the loss function
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': target_ids
        }


class T5NERDataset(Dataset):
    def __init__(self, tokenizer, data, input_column, target_column, source_max_len=128, target_max_len=128):
        self.tokenizer = tokenizer
        self.data_frame = data
        self.input_column = input_column
        self.target_column = target_column
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        # Retrieve the text and the labeled entities
        text = str(self.data_frame.iloc[index][self.input_column]) if not pd.isna(self.data_frame.iloc[index][self.input_column]) else "Missing text"
        ner_tags = str(self.data_frame.iloc[index][self.target_column]) if not pd.isna(self.data_frame.iloc[index][self.target_column]) else "Missing tags"

        # Prefix for NER task could be something like "ner: "
        t5_input = f"ner: {text}"

        # Tokenize the text
        tokenized_input = self.tokenizer(
            t5_input,
            max_length=self.source_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize the tags
        tokenized_target = self.tokenizer(
            ner_tags,
            max_length=self.target_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = tokenized_input['input_ids'].squeeze().to(dtype=torch.long)
        attention_mask = tokenized_input['attention_mask'].squeeze().to(dtype=torch.long)
        target_ids = tokenized_target['input_ids'].squeeze().to(dtype=torch.long)

        # Replace the padding token id's of the target by -100 so that it is ignored by the loss function
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': target_ids
        }



class T5TranslationDataset(Dataset):
    def __init__(self, tokenizer, data_frame, source_text_column, target_text_column, source_max_len=512, target_max_len=512):
        self.tokenizer = tokenizer
        self.data_frame = data_frame
        self.source_text = data_frame[source_text_column]
        self.target_text = data_frame[target_text_column]
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        source_text = str(self.source_text[index]) if pd.notna(self.source_text[index]) else "Missing data"
        target_text = str(self.target_text[index]) if pd.notna(self.target_text[index]) else "Missing data"

        # Prepare source text
        tokenized_source = self.tokenizer(
            source_text,
            max_length=self.source_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Prepare target text
        tokenized_target = self.tokenizer(
            target_text,
            max_length=self.target_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        source_ids = tokenized_source['input_ids'].squeeze()
        source_mask = tokenized_source['attention_mask'].squeeze()
        target_ids = tokenized_target['input_ids'].squeeze()
        
        # For training, T5 expects the decoder_input_ids to be the labels
        # which are to be shifted right, so ignore the first token of the target sequence
        labels = target_ids[1:].clone()
        decoder_input_ids = target_ids[:-1].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': source_ids.to(dtype=torch.long),
            'attention_mask': source_mask.to(dtype=torch.long),
            'decoder_input_ids': decoder_input_ids.to(dtype=torch.long),
            'labels': labels.to(dtype=torch.long)
        }

