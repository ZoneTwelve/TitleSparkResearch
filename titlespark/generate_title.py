from titlespark.utils import ChatCompletion, Conversation, Message
import asyncio
import fire
import re
import json
import random

def main(
    model: str = "gpt-3.5-turbo",
    api_base: str = "https://api.openai.com/v1",
    temperature: float = 0.7,
    num_threads: int = 1,
    topic_file: str = "categories.txt", 
    output: str = "titles.jsonl",
):
    api = ChatCompletion(api_base_url=api_base, model=model)
    parameters = {
        "temperature": temperature,
        "max_tokens": 250,
        #"seed": random.randint(0, 100000),
        "top_p": 0.98,
        "top_k": 5,
    }

    # Check if topic_file exists
    try:
        with open(topic_file, "r") as file:
            topics = file.read().splitlines()
    except FileNotFoundError:
        topics = ["human activity"]
        print(f"File {topic_file} not found.")
        return

    # Shuffle topics to ensure randomness
    random.shuffle(topics)

    used_topics = set()
    helpful_assistant = Message("system", "You follow the user's formatting requirements to the letter.")

    for _ in range(len(topics)):
        # Randomly select a topic that hasn’t been used yet
        available_topics = [topic for topic in topics if topic not in used_topics]
        if not available_topics:
            print("All topics have been used.")
            break

        topic = random.choice(available_topics)
        used_topics.add(topic)

        conversation = Conversation(
            [
                helpful_assistant,
                Message("user", f"根據以下主題產生 10 個相關的文章標題，格式為 <name>[title]</name>，請只回覆繁體中文：{topic}，禁止輸出 '{topic}' 的字串。"),
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

        # Regex pattern to capture title content between <title> and </title>
        pattern = r"<name>(.*?)</name>"

        # Initialize a list to hold titles
        all_titles = []

        # Loop through each response and extract titles
        for response in responses:
            print(f"Response: {response}")
            titles = re.findall(pattern, response)
            all_titles.extend(titles)

        # Print all extracted titles
        print(all_titles)

        # Append the titles to the JSON lines file, checking for duplicates
        with open(output, "a+") as file:
            file.seek(0)
            lines = file.readlines()
            existing_titles = set()
            for line in lines:
                existing_titles.add(json.loads(line)["title"])

            # Write unique titles to the file
            for title in all_titles:
                if title not in existing_titles:
                    file.write(json.dumps({"title": title, "source": topic}, ensure_ascii=False) + "\n")
                    print(f"Added title: {title} to titles.jsonl")

if __name__ == "__main__":
    fire.Fire(main)
