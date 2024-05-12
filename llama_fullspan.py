#!/usr/bin/python
# -*- coding: UTF-8 -*-
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
from metrics import match, loose_acc, metric_max_over_ground_truths, exact_match_score, f1_score, normalize_answer


seed = 633

torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def format_prompt_custom(prompt, evidences=None, instruction=None):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    
    if evidences is None:
        sys_prompt = B_SYS + "You are a helpful, respectful and honest assistant, Always answer as helpfully as possible. Your answers should only answer the question once and not have any text after the answer is done." + E_SYS
        ctxs = [""]
    else:
        sys_prompt = B_SYS + "You are a helpful, respectful and honest assistant, Always answer as helpfully as possible using the context text provided. Your answers should only answer the question once and not have any text after the answer is done." + E_SYS
        ctxs = []
        ctxs.append("Context: " + "\n".join([para["text"]for para in evidences]) + '\n\n')
        for i in evidences:
            ctxs.append("Context: " + i['text'] + '\n\n')
        
    if instruction is None:
        instruction = "Answer the question shortly and clearly." 
    instruction += '\n\n'
    prompt = f"Question: {prompt}"
    if "A, B, C and D" in instruction:
        # is multiple choice 
        return ['<s>' + B_INST + sys_prompt + i + instruction + prompt + E_INST + 'The best option is '  for i in ctxs]
    else:
        return ['<s>' + B_INST + sys_prompt + i + instruction + prompt + E_INST for i in ctxs]

    
def format_prompt_plain(prompt, evidences=None, instruction=None):
    if evidences is None:
        
        prompts = [prompt]
    else:
        
        ctxs = []
        ctxs.append('\n'.join(["{0}\n{1}\n".format(para["title"], para["text"]) for para in evidences]))
        for i in evidences:
            ctxs.append("{0}\n{1}\n".format(i["title"], i["text"]))
        prompts = ["{i}\n{p}".format(i=i, p=prompt) for i in ctxs]
    
    prompts = [[{"role": "user", "content": i}] for i in prompts]
    return prompts



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




def sequence_scoring(preds, evidences, rel_tokens=None, grd_tokens=None, ut_tokens=None,
                                     use_seqscore=False, w_rel=1.0, w_sup=1.0, w_use=0.5):
    results = {}

    overall_scores = {}
    for p_idx, pred in enumerate(preds):
        pred_token_ids = pred.outputs[0].token_ids
        pred_text = pred.outputs[0].text
        pred_id_log_probs = pred.outputs[0].id_log_probs

        seq_score = pred.outputs[0].cumulative_logprob / \
            max(len(pred_id_log_probs), 1)
        final_score = np.exp(seq_score)
        overall_scores[p_idx] = {"final_score": final_score}
        results["retrieval_{}".format(p_idx)] = {"pred": pred_text, "score": final_score, "id_log_probs": pred_id_log_probs, "token_ids": pred_token_ids, "evidence": evidences[p_idx]}
    return results

def call_model_rerank_w_scores_batch(prompt, evidences, model, score_method,
                                     ret_tokens=None, rel_tokens=None, grd_tokens=None, ut_tokens=None,
                                     use_seqscore=False, threshold=0.5,
                                     w_rel=1.0, w_sup=1.0, w_use=0.5, mode="adaptive_retrieval", closed=False, n_docs=5):
    final_results = {}
    prompt_no_ret, prompt_with_ret = prompt[0], prompt[1]
    prompt_use_all_ret = [prompt_with_ret[0]]
    prompt_use_one_ret = prompt_with_ret[1:]
    final_results["prompts"] = {"no_retrieval": prompt_no_ret, "all_doc_retrieval": prompt_use_all_ret, "one_doc_retrieval": prompt_use_one_ret}
    
    preds = model.generate(prompt_no_ret+prompt_use_all_ret+prompt_use_one_ret)
    
    assert len(preds) == 2+n_docs
    # index 0 is the no retrieval case
    final_results["no_retrieval"] = preds[0].outputs[0].text
    final_results["no_retrieval_ids"] = preds[0].outputs[0].token_ids
    final_results["no_retrieval_log_probs"] = preds[0].outputs[0].id_log_probs

    # index 1 is the all retrieval case
    final_results["all_doc_retrieval"] = preds[1].outputs[0].text
    final_results["all_doc_retrieval_ids"] = preds[1].outputs[0].token_ids
    final_results["all_doc_retrieval_log_probs"] = preds[1].outputs[0].id_log_probs

    # index 2: is the one retrieval case
    results = score_method(preds[2:], evidences, rel_tokens=rel_tokens, grd_tokens=grd_tokens, ut_tokens=ut_tokens,
                                    use_seqscore=use_seqscore, w_rel=w_rel, w_sup=w_sup, w_use=w_use)



    answer2score = {}
    if closed is True:
        for key, result in results.items():
            answer = postprocess_answer_option_conditioned(result["pred"])
            if len(answer.split()) > 0:
                answer = answer.split()[0]
            score = result["score"]
            answer2score.setdefault(answer, 0)
            answer2score[answer] += score
        sorted_answers = sorted(
            answer2score.items(), key=lambda x: x[1], reverse=True)
        best_option = sorted_answers[0][0]
        print(best_option)
        hit_results = {key: item for key, item in results.items() if postprocess_answer_option_conditioned(item["pred"]).startswith(best_option)}
        
        path2score = {key: item["score"] for key,
                        item in hit_results.items()}
        best_path = sorted(path2score.items(),
                            key=lambda x: x[1], reverse=True)[0][0]
        final_results["best_one"] = results[best_path]
        
        token_ids = results[best_path]["token_ids"]
        id_log_probs = results[best_path]["id_log_probs"]
    else:
        path2score = {key: item["score"] for key,
                        item in results.items()}
        best_path = sorted(path2score.items(),
                            key=lambda x: x[1], reverse=True)[0][0]
        best_option = results[best_path]["pred"]
        token_ids = results[best_path]["token_ids"]
        id_log_probs = results[best_path]["id_log_probs"]
        final_results["best_one"] = results[best_path]
    final_results["retrieval"] = best_option
    final_results["retrieval_token_ids"] = token_ids
    final_results["retrieval_log_probs"] = id_log_probs
    final_results["retrieval_res"] = results
    
    return final_results

def process_data_evidences(demonstration, top_n):
    ctx_key = "ctxs" if "ctxs" in demonstration else "top_contexts"
    prompt = PROMPT_DICT["prompt_no_input"].format_map(demonstration)
    evidences = demonstration[ctx_key][:top_n]
    return prompt, evidences

def preprocess_input_data(dataset, task=None):
    new_data = []
    
    if task in TASK_INST:
        instruction = TASK_INST[task]
    else:
        instruction = None
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
        else:
            item["instruction"] = item["question"]
        item["instruction"] = instruction + "\n\n" + item["instruction"] if instruction is not None else item["instruction"]
        new_data.append(item)

    return new_data
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--max_new_tokens', type=int, default=15)
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--download_dir', type=str, help="specify vllm model download dir",
                        default=".cache")
    parser.add_argument("--ndocs", type=int, default=5,
                        help="Number of documents to retrieve per questions")
    parser.add_argument("--world_size",  type=int, default=1,
                        help="world size to use multiple GPUs.")
    parser.add_argument("--dtype",  type=str, default="half",
                        help="We use bfloat16 for training. If you run inference on GPUs that do not support BF16, please set this to be `half`.")
    # Decoding hyperparams
    parser.add_argument('--threshold', type=float,
                        default=None, help="Adaptive threshold.")
    parser.add_argument("--use_seqscore", action="store_true")
    parser.add_argument("--use_groundness", action="store_true",
                        help="use ground score")
    parser.add_argument("--use_default_prompt", action="store_true",
                        help="use default prompt as selfrag")
    parser.add_argument(
        "--use_utility", action="store_true", help="tree search")
    parser.add_argument("--beam_width",  type=int,
                        default=2, help="beam search width")
    parser.add_argument("--max_depth",  type=int,
                        default=2, help="tree depth width")
    parser.add_argument("--w_rel",  type=float, default=1.0,
                        help="reward weight for document relevance")
    parser.add_argument("--w_sup",  type=float, default=1.0,
                        help="reward weight for generation support (attribution)")
    parser.add_argument("--w_use",  type=float, default=1.0,
                        help="reward weight for overall completeness / utility.")
    parser.add_argument('--mode', type=str, help="mode to control retrieval.",
                        default="default", choices=['adaptive_retrieval', 'no_retrieval', 'always_retrieve'],)
    parser.add_argument('--metric', type=str, help="metric to be used during evaluation")
    parser.add_argument('--few_shot', action="store_true")
    
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
    score_method = sequence_scoring
    # Get token ids for reflection tokens.
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=args.use_groundness, use_utility=args.use_utility)

    def generate(prompt, evidences):
        return call_model_rerank_w_scores_batch(prompt, evidences=evidences, model=model, score_method=score_method, 
                                                rel_tokens=rel_tokens, ret_tokens=ret_tokens, grd_tokens=grd_tokens, ut_tokens=ut_tokens,
                                                threshold=args.threshold, use_seqscore=args.use_seqscore,
                                                w_rel=args.w_rel, w_sup=args.w_sup, w_use=args.w_use, mode=args.mode, closed=args.task in ["fever", "arc_c"], n_docs=args.ndocs)

    
    all_results = []
    final_results = {}
    final_results["dataset"] = args.input_file
    acc = []
    match_score = []
    
    prompt_fn = None

    if args.use_default_prompt:
        print("Using default prompt")
        prompt_fn = format_prompt_plain
    else:
        print("Using custom prompt")
        prompt_fn = format_prompt_custom
    
    input_data = preprocess_input_data(input_data, task=args.task)
   
    few_shot_example = FEW_SHOT[args.task] if args.few_shot else ""
    few_shot_no_ret =  FEW_SHOT[f"{args.task}_no_ret"] if args.few_shot else ""
    for i, row in tqdm(enumerate(input_data)):
        
        _, evidences = process_data_evidences(row, top_n=args.ndocs)
        
        chats_no_ret = prompt_fn(prompt=row['instruction'], evidences=None, instruction=instruction)
        chats_with_ret = prompt_fn(prompt=row['instruction'], evidences=evidences, instruction=instruction)
        
        prompt_no_ret = [tokenizer.apply_chat_template(i, tokenize=False) for i in chats_no_ret]
        prompt_with_ret = [tokenizer.apply_chat_template(i, tokenize=False) for i in chats_with_ret]
        
        if args.few_shot:
            prompt_no_ret = [few_shot_no_ret + i for i in prompt_no_ret]
            prompt_with_ret = [few_shot_example + i for i in prompt_with_ret]
        
        if args.task == "arc_c":
            prompt_no_ret = [i + 'The best option is ' for i in prompt_no_ret]
            prompt_with_ret = [i + 'The best option is ' for i in prompt_with_ret]
        # if args.task == "fever":
        #     prompt_no_ret = [i + 'True or false? The statement is ' for i in prompt_no_ret]
        #     prompt_with_ret = [i + 'True or false? The statement is ' for i in prompt_with_ret] 
        res = generate([prompt_no_ret, prompt_with_ret], evidences)
        
        if 'id' in row:
            res['question_id'] = row['id']
        else:
            res['question_id'] = i # for pub health
    
        if "answers" not in row and "answer" in row:
            row["answers"] = [row["answer"]] if type(
                row["answer"]) is str else row["answer"]
        
        res['gold'] = row["answers"]
        
        all_results.append(res)
        pred = res["retrieval"]
        
        # print out first example
        if i == 0:
            print("=========Prompt no retrieval=========")
            print(prompt_no_ret[0])
            print("====Prompt with all doc retrieval====")
            print(prompt_with_ret[0])
            print("====Prompt with one doc retrieval====")
            for p in prompt_with_ret[1:]:
                print(p)
                print()
            print("===========Retrieval pred============")
            print(pred)
            print("=================Gold================")
            print(res["gold"])
        
        # if args.task == "fever":
        #     if "SUPPORTS" in pred:
        #         pred = "true"
        #     elif "REFUTES" in pred:
        #         pred = "false"
        if args.metric == "accuracy":
            acc.append(metric_max_over_ground_truths(loose_acc, pred, res["gold"]))
        else:
            match_score.append(match(pred, res["gold"]))
        if i % 10 == 0:
            if len(acc) > 0:
                print("acc: {}".format(np.mean(acc)))
            if len(match_score) > 0:
                print("match: {}".format(np.mean(match_score)))
            
    final_results["results"] = all_results
    if len(acc) > 0:
        print("acc: {}".format(np.mean(acc)))
    if len(match_score) > 0:
        print("match: {}".format(np.mean(match_score)))
    with open(args.output_file, "w") as outfile:
        json.dump(final_results, outfile)

if __name__ == "__main__":
    main()