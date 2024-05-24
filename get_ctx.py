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
from utils import FEW_SHOT, PROMPT_DICT, TASK_INST, load_jsonlines, control_tokens, load_special_tokens, save_file_jsonl
from metrics import loose_match, loose_acc, metric_max_over_ground_truths, exact_match_score, f1_score, normalize_answer
from datasets import Dataset
import pandas as pd
from torch.utils.data import DataLoader
from functools import singledispatch
from passage_retrieval import Retriever

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
MODEL_ROOT = os.environ.get("model_path")
WORK_ROOT = os.environ.get("work_path")

seed = 633

torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


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
            instruction = item["question"] + choices
        else:
            instruction = item["question"]
        
        new_data.append({'instruction': instruction})

    return new_data



     
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default=r'/apdcephfs_qy3/share_4983883/ping_test/rag/eval_data/hotpot_dev_fullwiki_v1.json')
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument("--ndocs", type=int, default=5,
                    help="Number of documents to retrieve per questions")
    
    args = parser.parse_args()
    input_path = args.input_file

    output_file = f"{input_path.split('.')[0]}_ctx.jsonl"
    if input_path.endswith(".json"):
        with open(input_path, 'r') as f:
            input_data = json.load(f)
    else:
        input_data = load_jsonlines(input_path)
    
    input_data_ = preprocess_input_data(input_data, task=args.task)
    
    dataset = Dataset.from_pandas(pd.DataFrame(input_data_))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    retriever = Retriever({})
    ret_model_path = f"{MODEL_ROOT}/contriever-msmarco" 
    passage_path = f"{WORK_ROOT}/retrieve_data/psgs_w100.tsv" 
    embedding_path = f"{WORK_ROOT}/retrieve_data/wikipedia_embeddings/*" 
    retriever.setup_retriever_demo(ret_model_path, passage_path, embedding_path,  n_docs=args.ndocs, save_or_load_index=True)
    
    for ind, batch in enumerate(dataloader):
        evidences = retriever.search_document_demo(batch['instruction'], args.ndocs)
        assert len(evidences) == len(batch['instruction'])
        assert len(evidences[0]) == args.ndocs
        for i, p in enumerate(evidences):
            input_data[ind*4+i]["ctxs"] = p

        if ind%100== 0:
            save_file_jsonl(input_data, output_file)
    save_file_jsonl(input_data, output_file)
    
if __name__ == "__main__":
    main()