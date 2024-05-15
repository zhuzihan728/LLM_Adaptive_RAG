
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from dotenv import load_dotenv, find_dotenv
import torch


load_dotenv(find_dotenv())
MODEL_ROOT = os.environ.get("model_path")

model_path = f"{MODEL_ROOT}/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"


tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
if device is None:
    device = torch.device("cpu")
model.to(device)

premise = "Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU"
hypothesis = "Emmanuel Macron is the President of France"

input = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
prediction = torch.softmax(output["logits"][0], -1).tolist()
label_names = ["entailment", "neutral", "contradiction"]
prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
print(prediction)