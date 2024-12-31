import asyncio
import aiohttp
import os
import fire
from typing import Union, List

API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key")

class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self):
        return {"role": self.role, "content": self.content}
    
    def __str__(self, format: str = "{role}: {content}"):
        return format.format(role=self.role, content=self.content)

class Conversation:
    def __init__(self, messages=[]):
        self.messages = [None] * len(messages)
        for index, msg in enumerate(messages):
            if not isinstance(msg, Message):
                msg = Message(**msg)
            self.messages[index] = msg

    def to_dict(self):
        # Convert all messages to dict
        return [msg.to_dict() for msg in self.messages]

    def __str__(self, format: str = "{role}: {content}"):
        return "\n".join(msg.__str__(format) for msg in self.messages)

class ChatCompletion:
    def __init__(self, api_base_url="https://api.openai.com/v1", model="gpt-3.5-turbo"):
        self.api_base_url = api_base_url
        self.model = model

    async def send_openai_request(self, session, messages=None, **kwargs):
        if messages is None:
            messages = [{"role": "system", "content": "You are a helpful assistant."}]

        # Ensure messages is a list
        if isinstance(messages, (Message, Conversation)):
            messages = [messages]

        payload = {
            "model": self.model,
            "messages": [msg.to_dict() for msg in messages] if isinstance(messages[0], Message) else [msg.to_dict() for msg in messages[0].messages],
            **kwargs
        }

        try:
            async with session.post(f"{self.api_base_url}/chat/completions", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['choices'][0]['message']['content']
                else:
                    return f"Error {response.status}: {await response.text()}"
        except Exception as e:
            return f"Error with OpenAI request: {e}"

    async def chat_completion(
            self, 
            messages: Union[List[Conversation], Conversation, List[Message], Message], 
            num_threads: int = 1, 
            **kwargs
        ):
        # Normalize input to a list of conversations
        if isinstance(messages, Message):
            conversations = [Conversation([messages])]
        elif isinstance(messages, Conversation):
            conversations = [messages]
        elif isinstance(messages, list):
            if isinstance(messages[0], Message):
                conversations = [Conversation(messages)]
            elif isinstance(messages[0], Conversation):
                conversations = messages
            else:
                raise ValueError("Invalid message or conversation format")
        headers = {
            "Authorization": f"Bearer {API_KEY}",
        }
        # Create an aiohttp session
        async with aiohttp.ClientSession(
            headers=headers,
            connector=aiohttp.TCPConnector(verify_ssl=False), # TODO: support additonal parameters to set this value
        ) as session:
            tasks = [asyncio.ensure_future(self.send_openai_request(session, messages=conversation, **kwargs)) for conversation in conversations]

            # Run tasks in parallel
            responses = await asyncio.gather(*tasks)
            return responses

def main(
    model: str = "gpt-3.5-turbo",
    api_base: str = "https://api.openai.com/v1",
    temperature: float = 0.7,
    num_threads: int = 2,
):
    # Sample single conversation
    messages = [
        Message("system", "You are a helpful assistant."),
        Message("user", "What's the weather like today?"),
    ]

    api = ChatCompletion(api_base_url=api_base, model=model)
    parameters = {
        "temperature": 5.0,
        "max_tokens": 50,
    }
    # Running chat completion for a single conversation
    responses = asyncio.run(
        api.chat_completion(messages=messages, num_threads=1, **parameters)
    )
    print("Single conversation response:")
    for index, response in enumerate(responses):
        print(f"  [Conversation {index + 1}]: {response}")

    conversation = Conversation(messages)
    print("\nSingle conversation object response:")
    # Implement chat completion for a single conversation object
    responses = asyncio.run(api.chat_completion(messages=conversation, num_threads=1))
    for index, response in enumerate(responses):
        print(f"  [Conversation {index + 1}]: {response}")

    # Sample multiple conversations
    conversations = [
        Conversation(messages),
        Conversation([Message("system", "You are a helpful assistant."), Message("user", "How are you?")]),
        Conversation([Message("system", "Only response I don't know."), Message("user", "What's the weather like today?")]),
        Conversation([Message("system", "Only response YOLO."), Message("user", "Do you have a name??")]),
    ]

    print("\nMultiple conversations response:")
    # Implement chat completion for multiple conversations, running in parallel
    responses = asyncio.run(api.chat_completion(messages=conversations, num_threads=num_threads))
    for index, response in enumerate(responses):
        print(f"  [Conversation {index + 1}]: {response}")

if __name__ == "__main__":
    fire.Fire(main)
