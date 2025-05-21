from transformers import BartForConditionalGeneration, BartTokenizer
import pandas as pd
import torch
from tqdm import tqdm

def exact_match(pred, gold):
    return pred.strip().lower() == gold.strip().lower()

def evaluate():
    model_dir = "models/bart-text2sql"

    tokenizer = BartTokenizer.from_pretrained(model_dir)
    model = BartForConditionalGeneration.from_pretrained(model_dir)
    model.eval()

    test = pd.read_parquet("data/raw/test.parquet")

    correct = 0
    total = len(test)

    for i in tqdm(range(10)):
        input_text = "Prompt: " + test['sql_prompt'].iloc[i] + ' Context: ' + test['sql_context'].iloc[i]
        true_sql = test['sql'].iloc[i]

        inputs = tokenizer(input_text, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=256,
                num_beams=4,
                early_stopping=True
            )
        pred_sql  = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if exact_match(pred_sql, true_sql):
            correct += 1

    accuracy = correct / total
    print(f"Exact Match Accuracy: {accuracy:.2%}")

if __name__ == '__main__':
    evaluate()