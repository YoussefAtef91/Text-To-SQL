import litserve as ls
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

class Text2SQL(ls.LitAPI):
    def setup(self, device="cpu"):
        model_dir = "models/bart-text2sql"
        self.tokenizer = BartTokenizer.from_pretrained(model_dir)
        self.model = BartForConditionalGeneration.from_pretrained(model_dir)
        self.model.eval()
        self.model.to(device)

    def predict(self, prompt: dict) -> dict:
        input_text = f"Prompt: {prompt['sql_prompt']} Context: {prompt['sql_context']}"
        inputs = self.tokenizer(input_text, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                num_beams=4,
                early_stopping=True
            )
        pred_sql = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"Generated Query": pred_sql}