import transformers
import torch
from tsfm_wrapper import MyModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
MODEL_ROOT = os.environ.get("model_path")

model_id = f"{MODEL_ROOT}/Meta-Llama-3-8B"

#pipeline = transformers.pipeline(pipeline = transformers.pipeline(
#    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"))
#pipeline("Hey how are you doing today?")

model_ = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
max_new_tokens = 50
model = MyModel(model_, tokenizer, max_new_tokens= max_new_tokens)

preds = model.generate(["Hello","Yummi is"])
for i in preds:
    print(i.outputs[0].text)
    print(i.outputs[0].token_ids)
    for j, token_id in enumerate(i.outputs[0].token_ids):
        print(token_id, "|", tokenizer.decode(token_id), "|", i.outputs[0].id_log_probs[j])
    print(i.outputs[0].id_log_probs)