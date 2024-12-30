import requests
from concurrent.futures import ThreadPoolExecutor
import re
import os
import json

BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "taide"
API_KEY = "your_api_key_here"  # Replace with your actual API key
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

DATASET_DIR = "datasets"

# Ensure dataset directory exists
os.makedirs(DATASET_DIR, exist_ok=True)

# Function to save progress to a JSONL file
def save_progress(dataset_name, data):
    file_path = os.path.join(DATASET_DIR, f"{dataset_name}.jsonl")
    with open(file_path, "a") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

# Function to load progress from a JSONL file
def load_progress(dataset_name):
    file_path = os.path.join(DATASET_DIR, f"{dataset_name}.jsonl")
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

# Function to make OpenAI chat completion API calls
def call_openai_chat_api(messages):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 200,
    }
    response = requests.post(f"{BASE_URL}/chat/completions", headers=HEADERS, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

# Task 1: Generate unique categories in the required format and verify with regex
def generate_categories(N):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Generate {N} unique categories in the format <category>$category</category>."}
    ]
    result = call_openai_chat_api(messages)
    categories = result.split("\n")
    pattern = re.compile(r"<category>.+</category>")
    verified_categories = [cat for cat in categories if pattern.match(cat)]
    return verified_categories

# Task 2: Generate an article title based on a category in the required format and verify with regex
def generate_article_title(category):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Based on the category '{category}', generate an article title in the format <title>$title</title>."}
    ]
    result = call_openai_chat_api(messages)
    pattern = re.compile(r"<title>.+</title>")
    if pattern.match(result):
        return result
    else:
        raise ValueError("Generated title does not match the required format.")

# Task 3: Generate an article directly from a category
def generate_article_from_category(category):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Based on the category '{category}', generate an article without any specific format."}
    ]
    return call_openai_chat_api(messages)

# Parallel execution handler
def parallel_generation(task_function, inputs, max_threads=5):
    results = []
    with ThreadPoolExecutor(max_threads) as executor:
        futures = [executor.submit(task_function, input_data) for input_data in inputs]
        for future in futures:
            results.append(future.result())
    return results

if __name__ == "__main__":
    dataset_name = "generated_dataset"

    # Load progress
    progress = load_progress(dataset_name)

    # Determine what needs to be generated
    existing_categories = [entry for entry in progress if entry.get("type") == "category"]
    existing_titles = [entry for entry in progress if entry.get("type") == "title"]
    existing_articles = [entry for entry in progress if entry.get("type") == "article"]

    # Step 1: Generate categories
    N = 10  # Number of categories to generate
    if len(existing_categories) < N:
        categories = generate_categories(N - len(existing_categories))
        save_progress(dataset_name, [{"type": "category", "content": category} for category in categories])
    else:
        categories = [entry["content"] for entry in existing_categories]

    # Step 2: Generate article titles based on categories (in parallel)
    M = 10  # Number of article titles to generate
    if len(existing_titles) < M:
        titles_to_generate = categories[:M - len(existing_titles)]
        titles = parallel_generation(generate_article_title, titles_to_generate, max_threads=5)
        save_progress(dataset_name, [{"type": "title", "content": title} for title in titles])
    else:
        titles = [entry["content"] for entry in existing_titles]

    # Step 3: Generate articles directly from categories (in parallel)
    if len(existing_articles) < M:
        articles_to_generate = categories[:M - len(existing_articles)]
        articles = parallel_generation(generate_article_from_category, articles_to_generate, max_threads=5)
        save_progress(dataset_name, [{"type": "article", "content": article} for article in articles])
    else:
        articles = [entry["content"] for entry in existing_articles]

    print("Categories:", categories)
    print("\nArticle Titles:", titles)
    print("\nArticles:", articles)
