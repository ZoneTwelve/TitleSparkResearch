from titlespark.utils import ChatCompletion, Conversation, Message
import asyncio
import fire
import re
import json
import random
import uuid

def main(
    model: str = "gpt-3.5-turbo",
    api_base: str = "https://api.openai.com/v1",
    temperature: float = 0.7,
    num_threads: int = 1,
    topic_file: str = "categories.txt",
    output: str = "output.jsonl",
):
    uuid4 = uuid.uuid4()
    api = ChatCompletion(api_base_url=api_base, model=model)
    parameters = {
        "temperature": temperature,
        "max_tokens": 800,
        "seed": random.randint(0, 100000),
        "top_p": 0.98,
        "top_k": 5,
    }

    # Load categories from existing file
    try:
        with open("categories.jsonl", "r") as file:
            print("Load categories")
            lines = file.readlines()
            all_categories = [json.loads(line)["category"] for line in lines]
    except FileNotFoundError:
        print("File categories.jsonl not found.")
        return

    # Shuffle and sample categories
    print("Shuffle and sample categories")
    random.shuffle(all_categories)
    length = [1,2,3,4,5]
    weights = [0.2, 0.4, 0.25, 0.1, 0.05]
    num = random.choices(length, weights=weights, k=1)[0]
    sample_categories = all_categories[:num]  # Sample 3 categories for this example

    helpful_assistant = Message("system", "You follow the user's formatting requirements to the letter.")

    # Generate article based on sampled categories
    selected_categories = ", ".join(sample_categories)
    conversation = Conversation([
        helpful_assistant,
        # Message("user", f"以下列格式產生一篇與 區塊鏈藝術, 飛行員, 網路滲透測試 相關的文章。文章必須只使用繁體中文，並以 `<content>[article]</content>` 格式輸出文章。"),
        # Message("assistant", f"<content>區塊鏈藝術結合創新技術，吸引飛行員興趣。網路滲透測試保障數位藝術安全，實現藝術與科技的完美融合，展現數位未來新方向。</content>"),
        Message("user", f"以下列格式產生一篇與 {selected_categories} 相關的文章。文章必須只使用繁體中文，並以 `<content>article context ... </content>` 格式輸出文章。"),
    ])
    print("\n[PROMPT]", conversation.__str__(format="<{role}>: {content}"))

    # Retrieve article from API
    print("Retrieve article from API")
    article_response = asyncio.run(
        api.chat_completion(
            messages=conversation,
            **parameters
        )
    )
    print("[RESPONSE]", article_response)

    # Extract the article content using regex
    article_pattern = r"<content>(.*?)</content>"
    article_match = re.search(article_pattern, article_response[0], re.DOTALL)
    article = article_match.group(1).strip() if article_match else ""

    # check if the article is empty
    if not article:
        print("article:", article, article_match)
        print("Article is empty. Skipping this example.")
        # save the result to a JSON file
        # save in {uuid4}.json
        raw_output = {
            "model": model,
            "api_base": api_base,
            "parameters": parameters,
            "conversation": conversation.to_dict(),
        }
        with open(f"outputs/{uuid4}-failed.json", "w") as file:
            file.write(json.dumps(raw_output, ensure_ascii=False) + "\n")
        print(f"Failed generation saved: {raw_output}")
        print("[[[ FAILED ]]]")
        return

    # Put the article content in the Conversation
    conversation.messages.append(Message("assistant", article_response[0]))
    conversation_gen_title = Conversation([
        helpful_assistant,
        Message("assistant", article_response[0]),
    ])

    # Generate a title for the article
    print("Generate a title for the article")
    conversation_gen_title.messages.append(Message("user", "根據文章內容生成一個適合的標題，並用繁體中文輸出，必須以 <name>title string...</name> 格式輸出。"))
    print("\n[PROMPT]", conversation_gen_title.__str__(format="<{role}>: {content}"))
    title_response = asyncio.run(
        api.chat_completion(
            messages=conversation_gen_title,
            **parameters
        )
    )
    print("[RESPONSE]", title_response)
    conversation_gen_title.messages.append(Message("assistant", title_response[0]))

    # Extract the title content using regex
    title_pattern = r"<name>(.*?)</name>"
    title_match = re.search(title_pattern, title_response[0], re.DOTALL)
    title = title_match.group(1).strip() if title_match else ""

    # Save the result to a JSON lines file
    raw_output = {
        "model": model if model != "taide" else "Aqua-mini",
        "api_base": api_base,
        "parameters": parameters,
        "conversations": [
            conversation.to_dict(),
            conversation_gen_title.to_dict(),
        ],
    }
    result = {
        "uuid": str(uuid4),
        "categories": sample_categories,
        "article": article,
        "title": title,
    }
    with open(f"outputs/{uuid4}-finish.json", "w") as file:
        file.write(json.dumps(raw_output, ensure_ascii=False) + "\n")

    with open(output, "a", encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Generated article and title saved: {result}")
    print(f"Save the raw output to {uuid4}-finish.json")
    print("\n\n\n{{{ FINISH }}}\n\n\n")

if __name__ == "__main__":
    fire.Fire(main)
