#!/usr/bin/env python3
"""
evaluation.py

Evaluate the trained mT5 model for both title generation and category generation.
Computes ROUGE for titles and a simple match metric for categories.
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_metric
import numpy as np


def generate_output(prompt_text, tokenizer, model, max_length=50):
    """
    Generate output (title or category) for a given prompt text.
    """
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    # 1. Load the model & tokenizer
    model_path = "final_model"  # Path where you saved your trained model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    
    # 2. Load the test dataset
    test_dataset = load_dataset(
        'json',
        data_files={'test': 'test.json'}
    )['test']
    
    # 3. Evaluate Title Generation with ROUGE
    rouge_metric = load_metric('rouge')
    
    all_pred_titles = []
    all_ref_titles = []
    
    for ex in test_dataset:
        article_text = ex['article']
        reference_title = ex['title']
        
        # Generate title
        prompt = f"title generation: {article_text}"
        predicted_title = generate_output(prompt, tokenizer, model, max_length=50)
        
        all_pred_titles.append(predicted_title)
        all_ref_titles.append(reference_title)
    
    # Compute ROUGE
    rouge_scores = rouge_metric.compute(predictions=all_pred_titles, references=all_ref_titles)
    print("Title Generation ROUGE Scores:", rouge_scores)
    
    # 4. Evaluate Category Generation with simple match or F1
    #    Here we do an example of how you might do an exact match or F1 for multi-label.
    #    Adjust as needed for your specific category metric requirements.
    
    # We'll do a simple exact match ratio here for illustration:
    correct = 0
    total = 0
    
    for ex in test_dataset:
        article_text = ex['article']
        true_categories = ex['categories']  # list of categories
        
        # Generate categories
        prompt = f"category generation: {article_text}"
        predicted_cats_str = generate_output(prompt, tokenizer, model, max_length=20)
        predicted_cats = [c.strip() for c in predicted_cats_str.split(",")]
        
        # Convert to sets for naive comparison
        true_cats_set = set([c.strip() for c in true_categories])
        pred_cats_set = set(predicted_cats)
        
        # Increment counters for exact set match
        if true_cats_set == pred_cats_set:
            correct += 1
        total += 1
    
    category_accuracy = correct / total if total > 0 else 0
    print(f"Category Generation Exact Match Accuracy: {category_accuracy:.4f}")


if __name__ == "__main__":
    main()

