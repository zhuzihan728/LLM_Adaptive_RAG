import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
MODEL_ROOT = os.environ.get("model_path")

model_path = f"{MODEL_ROOT}/Llama-2-7b-chat-hf"

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
chat = [
  {"role": "system", "content": "You are a helpful and honest assistant."},
  {"role": "user", "content": "How are you?"},
]
print(tokenizer.apply_chat_template(chat, tokenize=False))
