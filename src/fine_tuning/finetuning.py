from transformers import BartForConditionalGeneration, BartTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq

def finetune(tokenized_datasets):
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    training_args = TrainingArguments(
        output_dir="./bart-sql",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=10,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    print("Finished training")

    model_dir = "models/bart-text2sql"
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    print("Model and Tokenizer saved successfully.")


