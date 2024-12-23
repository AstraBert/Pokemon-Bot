import cohere
from typing import List, Dict
from dotenv import load_dotenv
import os

load_dotenv()

cohere_api_key = os.getenv("cohere_api_key")
co = cohere.ClientV2(cohere_api_key)

def chat_completion(message_history: List[Dict[str, str]]) -> str:
    response = co.chat(
        model="command-r-plus-08-2024",
        messages=message_history,
    ) 
    return response.message.content[0].text

def summarize(message, system_prompt="You are an helpful assistant whose job is to summarize the text you are given in less than 900 charachters (including spaces) in such a way that it would constitute an effective and engaging description of a pokemon card package"):
        response = chat_completion(
            message_history=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ]
        )
        return response