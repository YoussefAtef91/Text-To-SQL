import os
import subprocess
from datasets import load_from_disk
import litserve as ls
from src.fine_tuning.data_preparation import get_data, data_tokenization
from src.fine_tuning.finetuning import finetune
from src.deployment.api import Text2SQL

MODEL_PATH = "models/bart-text2sql"
TOKENIZED_PATH = "data/processed/train_tokenized"

def main():
    # If the model exists, launch the API
    if os.path.isdir(MODEL_PATH):
        print("Model found. Launching API...")
        api = Text2SQL()
        server = ls.LitServer(api, accelerator="cpu")
        server.run(port=8000, generate_client_file = False)
        return

    # If tokenized data exists, load it; otherwise, run data prep
    if os.path.isdir(TOKENIZED_PATH):
        print("Loading tokenized dataset...")
        tokenized_data = load_from_disk(TOKENIZED_PATH)
    else:
        print("Tokenized dataset not found. Running data preparation...")
        ds = get_data()
        tokenized_data = data_tokenization(ds)

    # Train the model
    print("Fine-tuning the model...")
    finetune(tokenized_data)

    # Launch the API
    print("Launching API...")
    api = Text2SQL()
    server = ls.LitServer(api, accelerator="cpu")
    server.run(port=8000, generate_client_file = False)

if __name__ == "__main__":
    main()
