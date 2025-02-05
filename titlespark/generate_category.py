from titlespark.utils import ChatCompletion, Conversation, Message
import asyncio
import fire
import re
import json
import random

def main(
    model: str = "gpt3.5-turbo",
    api_base: str = "https://api.openai.com/v1",
    temperature: float = 0.7,
    num_threads: int = 1,
    topic_file: str = "categories.txt",
    output: str = "output.jsonl",
):
    api = ChatCompletion(api_base_url=api_base, model=model)
    parameters = {
        "temperature": temperature,
        "max_tokens": 500,
        #"seed": random.randint(0, 100000),
        "top_p": 0.98,
        "top_k": 5,
    }

    # Load categories from existing file
    try:
        with open("categories.jsonl", "r") as file:
            lines = file.readlines()
            all_categories = [json.loads(line)["category"] for line in lines]
    except FileNotFoundError:
        print("File categories.jsonl not found.")
        return

    # Shuffle and sample categories
    random.shuffle(all_categories)
    sample_categories = all_categories[:3]  # Sample 3 categories for this example

    helpful_assistant = Message("system", "Ensure that all outputs strictly follow the user's requested format.")

    # Generate article based on sampled categories
    selected_categories = ", ".join(sample_categories)
    conversation = Conversation([
        helpful_assistant,
        Message("user", f"請生成一篇與以下主題相關的文章：{selected_categories}，文章需使用繁體中文，並且以 `<content>[article]</content>` 格式輸出文章。"),
    ])
    print(conversation.__str__(format="<{role}>: {content}"))

    # Retrieve article from API
    article_response = asyncio.run(
        api.chat_completion(
            messages=conversation,
            **parameters
        )
    )
    print(article_response)
    # Extract the article content using regex
    article_pattern = r"<content>(.*?)</content>"
    article_match = re.search(article_pattern, article_response[0])
    article = article_match.group(1).strip() if article_match else ""

    # Generate a title for the article
    conversation.messages.append(Message("user", "根據上述文章內容生成一個適當的標題，需為繁體中文標題並以 `<name>[title]</name>` 格式輸出標題。"))
    title_response = asyncio.run(
        api.chat_completion(
            messages=conversation,
            **parameters
        )
    )

    # Extract the title content using regex
    title_pattern = r"<name>(.*?)</name>"
    title_match = re.search(title_pattern, title_response[0])
    title = title_match.group(1).strip() if title_match else ""

    # Save the result to a JSON lines file
    result = {
        "categories": sample_categories,
        "article": f"<content>{article}</content>",
        "title": f"<name>{title}</name>",
    }

    with open(output, "a", encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Generated article and title saved: {result}")

if __name__ == "__main__":
    fire.Fire(main)
