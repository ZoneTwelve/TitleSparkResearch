#!/usr/bin/env python3
"""
inference.py

Run inference on new article text using the trained mT5 model.
Generates both a title and categories from the article text.
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def generate_text(prompt_text, tokenizer, model, max_length=50):
    """
    Helper function for generating text from a prompt.
    """
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_title(article_text, tokenizer, model):
    prompt = f"title generation: {article_text}"
    return generate_text(prompt, tokenizer, model, max_length=50)


def generate_categories(article_text, tokenizer, model):
    prompt = f"category generation: {article_text}"
    return generate_text(prompt, tokenizer, model, max_length=20)


def main():
    # 1. Load the model & tokenizer
    model_path = "final_model"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    
    # 2. An example article for inference
    article_text = (
        "Deep learning has revolutionized many fields in AI. "
        "Recent advances in transformers and attention mechanisms have driven "
        "progress in natural language processing, speech recognition, and computer vision."
    )
    
    # 3. Generate Title
    title = generate_title(article_text, tokenizer, model)
    print("Generated Title:", title)
    
    # 4. Generate Categories
    categories_str = generate_categories(article_text, tokenizer, model)
    print("Generated Categories:", categories_str)


if __name__ == "__main__":
    main()
