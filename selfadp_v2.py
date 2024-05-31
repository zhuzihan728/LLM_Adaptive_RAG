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



def preprocess_input_data(dataset, res, task=None):
    new_data = []
    assert len(dataset) == len(res)
    for ind, item in enumerate(dataset):
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
        elif task == "fever":
            item["instruction"] = f"Is the claim \"{item['question']}\" true or false?"
        else:
            item["instruction"] = item["question"]
        new_data.append({'instruction': f'Query: {item["instruction"]}\nAnswer: {res[ind]}'})

    return new_data

def get_retrieval_p(pred, tokenizer):
    global YES_TOKEN, NO_TOKEN
    has_judgment = False
    judge_ind = 0
    retrieve_p_hard = None
    for ind, id_ in enumerate(pred.outputs[0].token_ids):
        print(f"{id_}|{tokenizer.decode(id_)}")
        print(YES_TOKEN, NO_TOKEN)
        print(tokenizer.encode('Yes'), tokenizer.encode('No'))
        word = tokenizer.decode(id_).strip().lower()
        if id_ in [YES_TOKEN, NO_TOKEN]:
            has_judgment = True
            judge_ind = ind
            if word == 'yes':
                retrieve_p_hard = 1
            else:
                retrieve_p_hard = 0
            break
    log_prob_dc = pred.outputs[0].logprobs[judge_ind] # use the first token if no judgment
    retrieve_p = (np.exp(log_prob_dc[YES_TOKEN])/(np.exp(log_prob_dc[YES_TOKEN])+np.exp(log_prob_dc[NO_TOKEN])))
    if retrieve_p_hard is None:
        retrieve_p_hard = float(retrieve_p>0.5)
        
    probs = {'yes': np.exp(log_prob_dc[YES_TOKEN]), 'no': np.exp(log_prob_dc[NO_TOKEN])}
    return retrieve_p, retrieve_p_hard, has_judgment, probs

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
    parser.add_argument('--res_file', type=str)
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


    res_path = args.res_file
    with open(res_path, 'r') as f:
        res = json.load(f)
        res = res['results']
        res = [i['no_retrieval'] for i in res]
    
    model_ = AutoModelForCausalLM.from_pretrained(gpt, device_map='auto', torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(gpt, padding_side="left")
    max_new_tokens = args.max_new_tokens
    model = MyModel(model_, tokenizer, max_new_tokens=max_new_tokens)
    global YES_TOKEN, NO_TOKEN
    if "Llama-2" in args.model_name:
        YES_TOKEN = 8241
        NO_TOKEN = 3782
    else:
        YES_TOKEN = tokenizer.encode('Yes')[1]
        NO_TOKEN = tokenizer.encode('No')[1]
        
    input_data = preprocess_input_data(input_data, res, task=args.task)
    
    
    dataset = Dataset.from_pandas(pd.DataFrame(input_data))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # instruction = "Given the above user query, please make a judgment on whether you need some external documents from the web (e.g., Wikipedia) to correct answer it. Please answer yes or no."
    instruction = "Based on the given query and answer, do you think consulting external sources like Wikipedia would provide a more accurate answer? Please respond ONLY with #Yes or #No."
    
    res = {"retrieval_p":[], 'retrieval_p_hard':[], 'has_judgment':[], 'probs':[]}
    for ind, batch in enumerate(dataloader):
        prompts = [f"{i}\n\n{instruction}" for i in batch['instruction']]
        chats = [[{"role": "user", "content": i}] for i in prompts]
        if "Llama-3" in args.model_name:
            response_prefix = tokenizer.decode(128006) + tokenizer.decode(78191) + tokenizer.decode(128007) + tokenizer.decode(271)
            prompts = [tokenizer.apply_chat_template(chat, tokenize=False)+response_prefix for chat in chats]
            prompts = [i + '#' for i in prompts]
        elif "Llama-2" in args.model_name:
            prompts = [tokenizer.apply_chat_template(chat, tokenize=False) for chat in chats]
            prompts = [i + '#' for i in prompts]
        else:
            raise NotImplementedError
        
        pred = model.generate(prompts)
        
        for i, p in enumerate(pred):
            retrieve_p, retrieve_p_hard, has_judgment, probs = get_retrieval_p(p, tokenizer)
            res['retrieval_p'].append(retrieve_p)
            res['retrieval_p_hard'].append(retrieve_p_hard)
            res['has_judgment'].append(has_judgment)
            res['probs'].append(probs)
            print(f'===================== Batch {ind}, item {i} =====================')
            print(f"query: {prompts[i]}")
            print(f"judgment: {p.outputs[0].text}")
            print(f"retrieval_p: {retrieve_p}, retrieval_p_hard: {retrieve_p_hard}, has_judgment: {has_judgment}, probs: {probs}")
    
        if ind%100== 0:
            with open(args.output_file, 'w') as f:
                json.dump(res, f, default=to_serializable)
    with open(args.output_file, 'w') as f:
        json.dump(res, f, default=to_serializable)
    
    
if __name__ == "__main__":
    main()