from src.fine_tuning.data_preparation import get_data, data_tokenization
from src.fine_tuning.finetuning import finetune
from datasets import load_from_disk

def main():
    try:
        tokenized_data = load_from_disk("data/processed/train_tokenized")
    except:
        ds = get_data()
        tokenized_data = data_tokenization(ds)

    finetune(tokenized_data)

if __name__ == "__main__":
    main()