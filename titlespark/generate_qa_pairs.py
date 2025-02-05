from titlespark.utils import ChatCompletion, Conversation, Message
import asyncio
import fire
import re
import json
import random
import uuid
import pandas as pd

def main(
    model: str = "gpt-3.5-turbo",
    api_base: str = "https://api.openai.com/v1",
    temperature: float = 0.7,
    num_threads: int = 1,
    qa_file: str = "qa.csv",
    output: str = "output_qa_pairs.jsonl",
):
    qa_dataset = pd.read_csv(qa_file)
    uuid4 = uuid.uuid4()
    api = ChatCompletion(api_base_url=api_base, model=model)
    parameters = {
        "temperature": temperature,
        "max_tokens": 800,
        "seed": random.randint(0, 100000),
        "top_p": 0.98,
        "top_k": 5,
    }
    
    generated_questions = []

    for index, row in qa_dataset.iterrows():
        question = row["question"]

        helpful_assistant = Message("system", "You are an helpful assistant that follow the query that user asked.")

        # Generate article based on sampled categories
        conversation = Conversation([
            helpful_assistant,
            Message("user", f"提供與 '''<Q>{question}</Q>''' 十個類似的禮貌提問，並以 <Q>question content here...</Q> 輸出給使用者"),
        ])
        print("\n[PROMPT]", conversation.__str__(format="<{role}>: {content}"))

        # Retrieve article from API
        print("Retrieve article from API")
        response = asyncio.run(
            api.chat_completion(
                messages=conversation,
                **parameters
            )
        )

        question_pattern = r"<Q>(.*?)</Q>"
        question_match = re.findall(question_pattern, response[0], re.DOTALL)

        print("[RESPONSE]", response)
        print(question_match)

        generated_questions.append({"index": index, "generated_questions": question_match})

    question_dict = {item["index"]: item["generated_questions"] for item in generated_questions}
    qa_dataset['generated_questions'] = qa_dataset.index.map(question_dict)

    # Save the dataset as JSONL
    qa_dataset.to_json(output, orient='records', lines=True, force_ascii=False)


if __name__ == "__main__":
    fire.Fire(main)
