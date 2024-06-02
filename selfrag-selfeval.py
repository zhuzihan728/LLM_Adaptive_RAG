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
        assert len(res[ind]) == 5
        queries = '\n'.join([f"#{i+1}: {v}" for i, v in enumerate(res[ind])])
        new_data.append({'instruction': f'Query: {item["instruction"]}\nCandidate answers:\n{queries}'})

    return new_data

def get_scores(pred, tokenizer):
    global TOKEN_1, TOKEN_2, TOKEN_3, TOKEN_4, TOKEN_5
    has_selection = False
    selection_ind = 0
    select_hard = None
    for ind, id_ in enumerate(pred.outputs[0].token_ids):
        raw_word = tokenizer.decode(id_)
        print(f"target id: {id_}|{raw_word}|")
        print(TOKEN_1, TOKEN_2, TOKEN_3, TOKEN_4, TOKEN_5)
        word = tokenizer.decode(id_).strip().lower()
        if id_ in [TOKEN_1, TOKEN_2, TOKEN_3, TOKEN_4, TOKEN_5]:
            print(f"|{raw_word}|")
            has_selection = True
            selection_ind = ind
            select_hard = word
            break
    log_prob_dc = pred.outputs[0].logprobs[selection_ind] # use the first token if no judgment
    select_soft = {'1': np.exp(log_prob_dc[TOKEN_1]), '2': np.exp(log_prob_dc[TOKEN_2]), '3': np.exp(log_prob_dc[TOKEN_3]), '4': np.exp(log_prob_dc[TOKEN_4]), '5': np.exp(log_prob_dc[TOKEN_5])}
    if select_hard is None:
         select_hard = 'FAILED'
    return select_hard, select_soft, has_selection

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

    temp_res = []
    for i in res:
        temp = [v['pred'] for v in i['retrieval_res'].values()]
        temp_res.append(temp)
    res = temp_res
    
    model_ = AutoModelForCausalLM.from_pretrained(gpt, device_map='auto', torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(gpt, padding_side="left")
    max_new_tokens = args.max_new_tokens
    model = MyModel(model_, tokenizer, max_new_tokens=max_new_tokens)
    global TOKEN_1, TOKEN_2, TOKEN_3, TOKEN_4, TOKEN_5
    print('tokenized number 1-5:', tokenizer.encode("1"), tokenizer.encode("2"), tokenizer.encode("3"), tokenizer.encode("4"), tokenizer.encode("5"))
    if "Llama-2" in args.model_name:
        TOKEN_1, TOKEN_2, TOKEN_3, TOKEN_4, TOKEN_5 = tokenizer.encode("1")[2], tokenizer.encode("2")[2], tokenizer.encode("3")[2], tokenizer.encode("4")[2], tokenizer.encode("5")[2]
    else:
        TOKEN_1, TOKEN_2, TOKEN_3, TOKEN_4, TOKEN_5 = tokenizer.encode("1")[1], tokenizer.encode("2")[1], tokenizer.encode("3")[1], tokenizer.encode("4")[1], tokenizer.encode("5")[1]
    input_data = preprocess_input_data(input_data, res, task=args.task)
    
    
    dataset = Dataset.from_pandas(pd.DataFrame(input_data))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # instruction = "Given the above user query, please make a judgment on whether you need some external documents from the web (e.g., Wikipedia) to correct answer it. Please answer yes or no."
    instruction = "Given a query and five candidate answers, please select the most accurate answer to the query. Reply one of the following: #1, #2, #3, #4, #5. Then, explain your choice."
    
    res = {'prompts': [], 'answers': [], 'scores': []}
    for ind, batch in enumerate(dataloader):
        prompts = [f"{instruction}\n\n{i}" for i in batch['instruction']]
        chats = [[{"role": "user", "content": i}] for i in prompts]
        if "Llama-3" in args.model_name:
            response_prefix = tokenizer.decode(128006) + tokenizer.decode(78191) + tokenizer.decode(128007) + tokenizer.decode(271)
            prompts = [tokenizer.apply_chat_template(chat, tokenize=False)+response_prefix for chat in chats]
            prompts = [tokenizer.apply_chat_template(chat, tokenize=False) + 'The most accurate answer is #' for chat in chats]
        elif "Llama-2" in args.model_name:
            prompts = [tokenizer.apply_chat_template(chat, tokenize=False) for chat in chats]
            prompts = [tokenizer.apply_chat_template(chat, tokenize=False) + 'The most accurate answer is #' for chat in chats]
        elif 'selfrag' in args.model_name:
            template = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
            prompts = [template.format(instruction=instruction, input=i)+'The most accurate answer is #' for i in batch['instruction']]
        else:
            raise NotImplementedError
        pred = model.generate(prompts)
        
        for i, p in enumerate(pred):
            select_hard, select_soft, has_selection = get_scores(p, tokenizer)
            res['prompts'].append(prompts[i])
            res['answers'].append(select_hard)
            res['scores'].append(select_soft)
            
            print(f'===================== Batch {ind}, item {i} =====================')
            print(f"prompt: {prompts[i]}")
            print(f"selected: {select_hard}")
            print(f"scores: {select_soft}")
    
        if ind%100== 0:
            with open(args.output_file, 'w') as f:
                json.dump(res, f, default=to_serializable)
    with open(args.output_file, 'w') as f:
        json.dump(res, f, default=to_serializable)
    
    
if __name__ == "__main__":
    main()