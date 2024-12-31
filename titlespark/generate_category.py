from utils import ChatCompletion, Conversation, Message
import asyncio
import fire
import re
import json
import random

def main(
    model: str = "gpt-3.5-turbo",
    api_base: str = "https://api.openai.com",
    temperature: float = 0.7,
    num_threads: int = 2,
    topic_file: str = "categories.txt", 
):
    api = ChatCompletion(api_base_url=api_base, model=model)
    parameters = {
        "temperature": temperature,
        "max_tokens": 250,
        "seed": random.randint(0, 100000),
        # "use_beam_search": True,
        "top_p": 0.98,
        "top_k": 5,
    }

    # Check topic_file exist
    try:
        with open(topic_file, "r") as file:
            topics = file.read().splitlines()
    except FileNotFoundError:
        topics = ["human activity"]
        print(f"File {topic_file} not found.")
        return

    helpful_assistant = Message("system", "You follow the user's formatting requirements to the letter.")
    for topic in topics:
        conversation = Conversation(
            [
                helpful_assistant,
                Message("user", f"以下列格式產生 10 個類別 <name>[category]</name> 禁止輸出 '[category]' 字串 並且必須與 {topic} 有關，只回覆繁體中文。"),
            ]
        )
        print(conversation.__str__(format="<{role}>: {content}"))
        # Retrieve responses from API
        responses = asyncio.run(
            api.chat_completion(
                messages=conversation, 
                **parameters
            )
        )

        # Define the regex pattern to capture category content between <category> and </category>
        # pattern = r"<category>(.*?)</category>"
        pattern = r"<name>(.*?)</name>"

        # Initialize a list to hold all categories
        all_categories = []

        # Loop through each response and extract categories
        for response in responses:
            print(f"Response: {response}")
            categories = re.findall(pattern, response)
            all_categories.extend(categories)  # Flatten the categories into one list

        # Print all extracted categories
        print(all_categories)

        # Append the categories to the json lines file, checking for duplicates
        with open("categories.jsonl", "a+") as file:
            file.seek(0)
            lines = file.readlines()
            existing_categories = set()
            for line in lines:
                existing_categories.add(json.loads(line)["category"])

            # Write unique categories to the file
            for category in all_categories:
                if category not in existing_categories:
                    file.write(json.dumps({"category": category}, ensure_ascii=False) + "\n")
                    print(f"Added category: {category} to categories.jsonl")

if __name__ == "__main__":
    fire.Fire(main)
