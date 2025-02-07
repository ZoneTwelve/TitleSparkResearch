from titlespark.utils import ChatCompletion, Conversation, Message
from titlespark.prompt import artile_prompts_v2 as artile_prompts
import asyncio
import fire
import re
import json
import random
import concurrent.futures
from dotenv import load_dotenv
load_dotenv()

def generate_article(api, parameters, language, sample_categories, formatted_prompt):
    print("Generating article for", language)
    article_prompt = artile_prompts[language]
    prompt = article_prompt.format(LANGUAGE=language, PROMPT=formatted_prompt) + f"\nReference categories: {', '.join(sample_categories)}"
    conversation = Conversation([
        Message("system", "Ensure that all outputs strictly follow the user's requested format."),
        Message("user", prompt),
    ])
    print(conversation.__str__(format="<{role}>: {content}"))
    
    # Retrieve article from API
    article_response = asyncio.run(
        api.chat_completion(
            messages=conversation,
            **parameters
        )
    )
    # print(article_response)
    
    # Extract the article content using regex
    article_pattern = r"\[story\](.*?)\[/story\]"
    article_match = re.search(article_pattern, article_response[0], re.DOTALL)

    article = article_match.group(1).strip() if article_match else article_response[0]
    print("Article:")
    print("\tArticle response:", article_response[0])
    print("\n")
    print("\tArticle extract:", article)
    print("\n\n")
    # Return the result instead of writing to a file directly
    return {
        "categories": sample_categories,
        "article": article,
        "language": language,
        "validate": True if article_match else False
    }

def main(
    model: str = "gpt3.5-turbo",
    api_base: str = "https://api.openai.com/v1",
    temperature: float = 0.7,
    num_threads: int = 1,
    topic_file: str = "categories.txt",
    output: str = "multilingual.jsonl",
):
    api = ChatCompletion(api_base_url=api_base, model=model)
    parameters = {
        "temperature": temperature,
        "max_tokens": 1200,
        "top_p": 0.7,
        "top_k": 2,
    }
    formatted_prompt = "[story]article content[/story]"
    
    # Load categories from existing file
    print("Loading categories from file...")
    try:
        with open(topic_file, "r") as file:
            lines = file.readlines()
            all_categories = [line.strip() for line in lines]
    except FileNotFoundError:
        print("File categories.jsonl not found.")
        return
    
    # Shuffle and sample categories
    print("Sampling categories...")
    random.shuffle(all_categories)
    # deduplicate categories
    all_categories = list(set(all_categories))
    random_length = random.randint(1, 5)
    sample_categories = all_categories[:random_length]
    print("Sampled categories:", sample_categories)
    # Multithreading to process multiple languages in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(generate_article, api, parameters, language, sample_categories, formatted_prompt): language
            for language in artile_prompts
        }
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            res['model'] = model
            results.append(res)
    
    # Write all results to file after threads complete

    print("Writing results to file...")
    with open(output, "a", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"Generated articles saved to {output}")

if __name__ == "__main__":
    fire.Fire(main)