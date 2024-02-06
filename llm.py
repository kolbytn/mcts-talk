from typing import List, Dict
import os
import time
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]


def get_response(messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo-0125", max_tries=50, temperature=1, stop=["\n", "("], **kwargs) -> str:
    completion = None
    num_tries = 0
    while not completion and num_tries < max_tries:
        try:
            completion = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stop=stop,
                **kwargs
            ).choices[0].message.content
            break
        except Exception as e:
            num_tries += 1
            print("try {}: {}".format(num_tries, e))
            if "maximum context length" in str(e):
                if len(messages) > 3:
                    if messages[0]["role"] == "system":
                        messages = [messages[0]] + messages[3:]
                    else:
                        messages = messages[2:]
                else:
                    raise RuntimeError("messages too long")
            time.sleep(2)
    if not completion:
        raise RuntimeError("Failed to get response from API")
    return completion
