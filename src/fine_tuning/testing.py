from transformers import BartForConditionalGeneration, BartTokenizer
import pandas as pd

def test():
    model_dir = "models/bart-text2sql"

    tokenizer = BartTokenizer.from_pretrained(model_dir)
    model = BartForConditionalGeneration.from_pretrained(model_dir)

    test = pd.read_parquet("data/raw/test.parquet")

    for i in range(5):
        input_text = "Prompt: " + test['sql_prompt'].iloc[i] + ' Context: ' + test['sql_context'].iloc[i]
        true_output = test['sql'].iloc[i]

        inputs = tokenizer(input_text, return_tensors="pt")

        outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"Generated Query: {decoded_output}")
        print(f"True Query: {true_output}")
        print("------------------")

if __name__ == '__main__':
    test()