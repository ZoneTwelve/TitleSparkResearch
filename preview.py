import json
from transformers import T5Tokenizer

def preview_tokenization(dataset_path, model_name="t5-small", max_input_length=512, max_output_length=128, sample_size=5):
    # Load the tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # Load dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # Limit to sample size for preview
    samples = data[:sample_size]

    for idx, sample in enumerate(samples):
        article = sample["article"]
        title = sample["title"]

        # Tokenize article and title
        article_tokens = tokenizer(
            article, truncation=True, padding="max_length", max_length=max_input_length
        )
        title_tokens = tokenizer(
            title, truncation=True, padding="max_length", max_length=max_output_length
        )

        # Detokenize to verify correctness
        detokenized_article = tokenizer.decode(article_tokens["input_ids"], skip_special_tokens=True)
        detokenized_title = tokenizer.decode(title_tokens["input_ids"], skip_special_tokens=True)

        # Show results
        print(f"Sample {idx + 1}:")
        print("Original Article:", article)
        print("Tokenized Article Input IDs:", article_tokens["input_ids"])
        print("Detokenized Article:", detokenized_article)
        print("\nOriginal Title:", title)
        print("Tokenized Title Input IDs:", title_tokens["input_ids"])
        print("Detokenized Title:", detokenized_title)
        print("=" * 50)

if __name__ == "__main__":
    # Path to the dataset
    dataset_path = "dataset.jsonl"
    model_name = "google/mt5-small"
    # Preview tokenization
    preview_tokenization(dataset_path, model_name=model_name)

