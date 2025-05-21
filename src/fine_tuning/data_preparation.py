import pandas as pd
from datasets import Dataset
from transformers import BartTokenizer


def get_data():
    splits = {'train': 'synthetic_text_to_sql_train.snappy.parquet', 'test': 'synthetic_text_to_sql_test.snappy.parquet'}
    train = pd.read_parquet("hf://datasets/gretelai/synthetic_text_to_sql/" + splits["train"])
    test = pd.read_parquet("hf://datasets/gretelai/synthetic_text_to_sql/" + splits["test"])

    columns = ['sql_prompt', 'sql_context', 'sql']

    train = train[columns]
    test = test[columns]

    train.to_parquet("data/raw/train.parquet")
    test.to_parquet("data/raw/test.parquet")

    train['input_text'] = 'Prompt: ' + train['sql_prompt'] + ' Context: ' + train['sql_context']
    train['target_text'] = train['sql']

    train[['input_text','target_text']].to_parquet("data/processed/train_processed.parquet")

    dataset = Dataset.from_pandas(train[['input_text','target_text']])
    dataset = dataset.train_test_split(test_size=0.1)

    return dataset
    


def data_tokenization(dataset):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    def preprocess(example):
        model_input = tokenizer(
            example["input_text"],
            max_length=128,
            padding="max_length",
            truncation=True,)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                example["target_text"],
                max_length=128,
                padding="max_length",
                truncation=True,)
        model_input["labels"] = labels["input_ids"]
        return model_input

    tokenized_dataset = dataset.map(preprocess, batched=True)

    tokenized_dataset.save_to_disk("data/processed/train_tokenized")
    return tokenized_dataset



if __name__ == "__main__":
    ds = get_data()
    tokenized_data = data_tokenization(ds)

