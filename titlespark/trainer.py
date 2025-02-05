#!/usr/bin/env python3
"""
trainer.py

Script to train an mT5 model (via AutoModelForSeq2SeqLM) on title/category generation.
"""

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset


def preprocess_function(examples, tokenizer):
    """
    Preprocess function to create input-output pairs for both title and category generation
    based on multi-task prompts.
    """
    inputs = []
    targets = []
    
    for article, title, categories in zip(examples['article'], examples['title'], examples['categories']):
        # For TITLE generation
        title_input = f"title generation: {article}"
        title_output = title
        inputs.append(title_input)
        targets.append(title_output)
        
        # For CATEGORY generation
        category_input = f"category generation: {article}"
        category_output = ", ".join(categories)  # Convert list to comma-separated string
        inputs.append(category_input)
        targets.append(category_output)
    
    # Tokenize inputs
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    
    # Tokenize targets (labels)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    # 1. Load dataset
    raw_datasets = load_dataset(
        'json',
        data_files={
            'train': 'train.json',
            'validation': 'val.json',
            'test': 'test.json'
        }
    )
    
    # 2. Initialize tokenizer and model
    model_name = "google/mt5-base"  # or any other mT5 checkpoint
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # 3. Preprocess dataset
    def tokenize_function(examples):
        return preprocess_function(examples, tokenizer)
    
    # Use .map() to tokenize the dataset
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names
    )
    
    # 4. Training arguments
    training_args = TrainingArguments(
        output_dir="output",
        evaluation_strategy="steps",  # or "epoch"
        eval_steps=500,
        save_steps=500,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        logging_steps=100,
        logging_dir="logs",
        report_to="tensorboard",
        load_best_model_at_end=True,
        save_total_limit=2  # limit the number of saved checkpoints
    )
    
    # 5. Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"]
    )
    
    # 6. Train
    trainer.train()
    
    # 7. Save the final model
    trainer.save_model("final_model")
    tokenizer.save_pretrained("final_model")


if __name__ == "__main__":
    main()

