import spacy
import jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM
from tsfm_wrapper import MyModel
import random
import torch
import os
import numpy as np
import openai
from tqdm import tqdm
import json
import argparse
import ast
import re
from tqdm import tqdm
from collections import Counter
import string
import sys
import time
from utils import FEW_SHOT, PROMPT_DICT, TASK_INST, load_jsonlines, control_tokens, load_special_tokens
from metrics import loose_match, loose_acc, metric_max_over_ground_truths, exact_match_score, f1_score, normalize_answer
from datasets import Dataset
import pandas as pd
from torch.utils.data import DataLoader
from functools import singledispatch
seed = 633

torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

YES_TOKEN = None
NO_TOKEN = None
def postprocess_answer_option_conditioned(answer):
    for token in control_tokens:
        answer = answer.replace(token, "")

    if "</s>" in answer:
        answer = answer.replace("</s>", "")
    if "\n" in answer:
        answer = answer.replace("\n", "")

    if "<|endoftext|>" in answer:
        answer = answer.replace("<|endoftext|>", "")
    if type(answer) is str and len(answer) > 0 and (answer[0] == "#" or answer[0] == ":"):
        answer = answer[1:]
    return normalize_answer(answer)


def preprocess_input_data(dataset, task=None):
    new_data = []
    for item in dataset:
        if task == "arc_c":
            choices = item["choices"]
            answer_labels = {}
            for i in range(len(choices["label"])):
                answer_key = choices["label"][i]
                text = choices["text"][i]
                if answer_key == "1":
                    answer_labels["A"] = text
                if answer_key == "2":
                    answer_labels["B"] = text
                if answer_key == "3":
                    answer_labels["C"] = text
                if answer_key == "4":
                    answer_labels["D"] = text
                if answer_key in ["A", "B", "C", "D"]:
                    answer_labels[answer_key] = text

            if "D" not in answer_labels:
                answer_labels["D"] = ""
            choices = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(
                answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
            if "E" in answer_labels:
                choices += "\nE: {}".format(answer_labels["E"])
            item["instruction"] = item["question"] + choices
            item["answers"] = [item["answerKey"]]
        elif task == "fever":
            item["instruction"] = f"Is the claim \"{item['question']}\" true or false?"
        else:
            item["instruction"] = item["question"]
        
        new_data.append({'instruction': item["instruction"]})

    return new_data

def get_retrieval_p(pred, tokenizer):
    has_judgment = False
    judge_ind = 0
    retrieve_p_hard = None
    for ind, id_ in enumerate(pred.outputs[0].token_ids):
        word = tokenizer.decode(id_).strip().lower()
        if word == 'yes' or word == 'no':
            has_judgment = True
            judge_ind = ind
            if word == 'yes':
                retrieve_p_hard = 1
            else:
                retrieve_p_hard = 0
            break
    log_prob_dc = pred.outputs[0].logprobs[judge_ind] # use the first token if no judgment
    global YES_TOKEN, NO_TOKEN
    retrieve_p = (np.exp(log_prob_dc[YES_TOKEN])/(np.exp(log_prob_dc[YES_TOKEN])+np.exp(log_prob_dc[NO_TOKEN])))
    if retrieve_p_hard is None:
        retrieve_p_hard = float(retrieve_p>0.5)
    return retrieve_p, retrieve_p_hard, has_judgment

@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)

@to_serializable.register(np.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return np.float64(val)
     
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--max_new_tokens', type=int, default=15)
   
    
    args = parser.parse_args()
    gpt = args.model_name
    input_path = args.input_file
    if input_path.endswith(".json"):
        input_data = json.load(open(input_path))
    else:
        input_data = load_jsonlines(input_path)

    if args.task in TASK_INST:
        instruction = TASK_INST[args.task]
    else:
        instruction = None
        
    model_ = AutoModelForCausalLM.from_pretrained(gpt, device_map='auto', torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(gpt, padding_side="left")
    max_new_tokens = args.max_new_tokens
    model = MyModel(model_, tokenizer, max_new_tokens=max_new_tokens)
    
    res={"retrieval_p":[], 'retrieval_p_hard':[], 'has_judgment':[], 'output':[]}
    
    input_data = preprocess_input_data(input_data, task=args.task)
    
    
    dataset = Dataset.from_pandas(pd.DataFrame(input_data))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    instruction = "Given the above user query, please make a judgment on whether finding some external documents from the web (e.g., Wikipedia) helps to generate a better response. Please answer yes or no."
    
    global YES_TOKEN, NO_TOKEN
    YES_TOKEN = tokenizer.encode('Yes')[1]
    NO_TOKEN = tokenizer.encode('No')[1]
    
    
    for ind, batch in enumerate(dataloader):
        prompts = [f"Query: {i}\n\n{instruction}" for i in batch['instruction']]
        chats = [[{"role": "user", "content": i}] for i in prompts]
        prompts = [tokenizer.apply_chat_template(chat, tokenize=False)+'assistant' for chat in chats]
        pred = model.generate(prompts)
        
        for i, p in enumerate(pred):
            retrieve_p, retrieve_p_hard, has_judgment = get_retrieval_p(p, tokenizer)
            res['retrieval_p'].append(retrieve_p)
            res['retrieval_p_hard'].append(retrieve_p_hard)
            res['has_judgment'].append(has_judgment)
            res['output'].append({'text':p.outputs[0].text, 'token_ids':p.outputs[0].token_ids, 'logprobs':p.outputs[0].logprobs, 'id_log_probs':p.outputs[0].id_log_probs})
            print(f'===================== Batch {ind}, item {i} =====================')
            print(f"query: {batch['instruction'][i]}")
            print(f"judgment: {p.outputs[0].text}")
            print(f"retrieval_p: {retrieve_p}, retrieval_p_hard: {retrieve_p_hard}, has_judgment: {has_judgment}")
    

    with open(args.output_file, 'w') as f:
        json.dump(res, f, default=to_serializable)

    # for i, row in tqdm(enumerate(input_data)):
    #     prompt = [f"Query: {row['instruction']}\n\n{instruction}"]
    #     chats = [[{"role": "user", "content": i}] for i in prompt]
    #     prompts = [tokenizer.apply_chat_template(chat, tokenize=False)+'assistant\n\n' for chat in chats]
    #     print('===================== prompt =====================')
    #     for p in prompts:
    #         print(p)
    #     pred = model.generate(prompts)
    #     print('===================== pred =====================')
    #     print(pred[0].outputs[0].text)
    #     print(pred[0].outputs[0].token_ids)
    #     for id_ in pred[0].outputs[0].token_ids:
    #         print(id_, '|', tokenizer.decode(id_))
    #     print(tokenizer.encode('yes'), tokenizer.encode('no'))
    #     if i >= 3:
    #         break
    
    
if __name__ == "__main__":
    main()