import json
from transformers import PreTrainedTokenizer

def load_dataset(file_path):
    """
    Load a JSONL dataset into a list of dictionaries.

    Args:
        file_path (str): Path to the dataset file.

    Returns:
        list: List of dictionaries with `article` and `title` keys.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = [json.loads(line) for line in file]
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {file_path}")
        return []

def split_dataset(data, test_size=0.2):
    """
    Split a dataset into training and validation subsets.

    Args:
        data (list): List of dataset entries.
        test_size (float): Proportion of data to use for validation.

    Returns:
        tuple: (train_data, val_data)
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(data, test_size=test_size, random_state=42)

def tokenize_dataset(data, tokenizer: PreTrainedTokenizer, max_input_length=512, max_output_length=128):
    """
    Tokenize a dataset of articles and titles.

    Args:
        data (list): List of dictionaries with `article` and `title` keys.
        tokenizer (PreTrainedTokenizer): Pretrained tokenizer.
        max_input_length (int): Max length for input sequences.
        max_output_length (int): Max length for output sequences.

    Returns:
        list: List of tokenized inputs and labels.
    """
    tokenized_data = []
    for entry in data:
        article = entry["article"]
        title = entry["title"]

        input_encoding = tokenizer(
            article, truncation=True, padding="max_length", max_length=max_input_length, return_tensors="pt"
        )
        target_encoding = tokenizer(
            title, truncation=True, padding="max_length", max_length=max_output_length, return_tensors="pt"
        )

        tokenized_data.append({
            "input_ids": input_encoding["input_ids"].squeeze(0),
            "attention_mask": input_encoding["attention_mask"].squeeze(0),
            "labels": target_encoding["input_ids"].squeeze(0),
        })

    return tokenized_data

def deduplicate_dataset(data):
    """
    Remove duplicate entries from the dataset.

    Args:
        data (list): List of dataset entries.

    Returns:
        list: Deduplicated dataset.
    """
    seen = set()
    deduplicated_data = []
    for entry in data:
        entry_tuple = (entry["article"], entry["title"])
        if entry_tuple not in seen:
            deduplicated_data.append(entry)
            seen.add(entry_tuple)
    return deduplicated_data

def save_dataset(data, file_path):
    """
    Save a dataset to a JSONL file.

    Args:
        data (list): List of dataset entries.
        file_path (str): Path to save the dataset.

    Returns:
        None
    """
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            for entry in data:
                file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Dataset saved to {file_path}")
    except Exception as e:
        print(f"Error saving dataset: {e}")

if __name__ == "__main__":
    # Example usage
    dataset_path = "dataset.jsonl"
    tokenizer_name = "google/mt5-small"
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Load dataset
    data = load_dataset(dataset_path)
    print(f"Loaded {len(data)} entries.")

    # Deduplicate dataset
    data = deduplicate_dataset(data)
    print(f"Deduplicated dataset contains {len(data)} entries.")

    # Tokenize dataset
    tokenized_data = tokenize_dataset(data, tokenizer)
    print(f"Tokenized {len(tokenized_data)} entries.")
