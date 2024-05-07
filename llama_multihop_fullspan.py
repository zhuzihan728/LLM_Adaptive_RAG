#!/usr/bin/python
# -*- coding: UTF-8 -*-

import spacy
import jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM
# from tsfm_wrapper import MyModel
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
from utils import PROMPT_DICT, TASK_INST, load_jsonlines, control_tokens, load_special_tokens
from metrics import match, loose_acc, metric_max_over_ground_truths, exact_match_score, f1_score
from passage_retrieval import Retriever
from tsfm_wrapper import MyModel
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

TEST_MODEL_NAME = None


def postprocess_answer_option_conditioned(answer):
    for token in control_tokens:
        answer = answer.replace(token, "")

    if "</s>" in answer:
        answer = answer.replace("</s>", "")
    if "\n" in answer:
        answer = answer.replace("\n", "")

    if "<|endoftext|>" in answer:
        answer = answer.replace("<|endoftext|>", "")
    if type(answer) is str and (answer[0] == "#" or answer[0] == ":"):
        answer = answer[1:]
    return answer

TEST_MODEL_NAME = "llama2-7b"

# def llama_format_prompt(prompt, evidences=None, instruction=None, prev_evidence=[], prev_a = "", use_all_evidences=False):
#     B_INST, E_INST = "[INST]", "[/INST]"
#     B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    
#     if evidences is None and len(prev_evidence) == 0:
#         # no evidences
#         sys_prompt = B_SYS + "You are a helpful, respectful and honest assistant, Always answer as faithful and truthful as possible." + E_SYS
#         ctxs = [""]
#     else:
#         sys_prompt = B_SYS + "You are a helpful, respectful and honest assistant, Always answer as faithful and truthful as possible using the context text provided." + E_SYS
#         prev_evidence_text = "\n".join([para["text"]for para in prev_evidence]) + '\n'
#         ctxs = []
#         if use_all_evidences:
#             # add all new evidences
#             ctxs.append("Context: " + prev_evidence_text + "\n".join([para["text"]for para in evidences]) + '\n\n')
#         else:
#             # add one new evidence for each prompt
#             for i in evidences:
#                 ctxs.append("Context: " + prev_evidence_text + i['text'] + '\n\n')
        
#     if instruction is None:
#         instruction = "Answer the question shortly and clearly." 
#     instruction += '\n\n'
#     prompt = f"Question: {prompt}"
    
#     return ['<s>' + B_INST + sys_prompt + i + instruction + prompt + E_INST + prev_a for i in ctxs]

def llama_template(prompt, evidences=None, instruction=None, prev_chat=[], use_all_evidences=False):
    """
        prompt (str): the question, or previous answer
        evidences (List(Dict)): a list of evidence paragraphs
        instruction (str, optional): task instruction. Only used in the first turn. Defaults to None.
        prev_chat (List(Dict)): the best chat history. Defaults to [].
        use_all_evidences (bool, optional): whether to use all evidences. Defaults to False.
    """
    ctxs = []
    if use_all_evidences:
        ctxs.append("\n".join([para["text"]for para in evidences]))
    else:
        for i in evidences:
            ctxs.append(i['text'])
    
    
    
    if len(prev_chat) == 0:
        # initiate chat
        print("initiate chat")
        new_chats = []
        instruction = "" if instruction is None else ' ' + instruction
        for ctx in ctxs:
            chat = [
                {"role": "system", "content": f"You are a helpful, respectful and honest assistant, Always answer as faithful and truthful as possible using the context text provided.{instruction}"},
                {"role": "user", "content": f"Context: {ctx}\n\nQuestion: {prompt}"},
            ]
            new_chats.append(chat)
    else:
        new_chats = []
        for ctx in ctxs:
            chat = prev_chat.copy()
            
            chat.append({"role": "assistant", "content": prompt})
            chat.append({"role": "user", "content": f"Added context: {ctx}\n\nRefine your answer:"})
            new_chats.append(chat)
    
    return new_chats

def selfrag_format_prompt(prompt, evidences=None, instruction=None, prev_prompt=None):
    prompt_no_input =  "### Instruction:\n{instruction}\n\n### Response:\n"
    
    if prev_prompt is None:
        if instruction is not None:
            instruction = instruction + "\n\n### Input:\n" + prompt
        else:
            instruction = prompt
        prompt = prompt_no_input.format(instruction=instruction)
        return prompt
    
    
    else:
        prompt = prev_prompt+ "".join(["[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(
            para["title"], para["text"]) for para in evidences])+prompt
        return prompt
    
def decide_retrieval(pred_log_probs, pred, ret_tokens, threshold):
    # return the probability of retrieval
    if threshold is not None:
        score_dict = {}
        for tok, id in ret_tokens.items():
            if id not in pred_log_probs[0]:
                score_dict[tok] = -100
            prob = pred_log_probs[0][id]
            score_dict[tok] = float(prob)
        do_retrieve = score_dict["[Retrieval]"] / (
            score_dict["[Retrieval]"] + score_dict["[No Retrieval]"]) > threshold
    else:
        do_retrieve = float("[Retrieval]" in pred)
    return do_retrieve



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
    return answer



def sequence_scoring(preds, rel_tokens=None, grd_tokens=None, ut_tokens=None,
                                     use_seqscore=False, w_rel=1.0, w_sup=1.0, w_use=0.5, prompts = None, evidences=None):
    results = {}

    overall_scores = {}
    for p_idx, pred in enumerate(preds):
        pred_token_ids = pred.outputs[0].token_ids
        pred_text = pred.outputs[0].text
        pred_id_log_probs = pred.outputs[0].id_log_probs

        seq_score = pred.outputs[0].cumulative_logprob / \
            max(len(pred.outputs[0].token_ids), 1)
        final_score = np.exp(seq_score)
        overall_scores[p_idx] = {"final_score": final_score}
        results["retrieval_{}".format(p_idx)] = {"pred": pred_text, "score": final_score, "id_log_probs": pred_id_log_probs, "token_ids": pred_token_ids, "evidences": [evidences[p_idx]], "template": prompts[p_idx]}
    return results

def call_model_rerank_w_scores_batch(prompt, evidences, model, score_method,
                                     ret_tokens=None, rel_tokens=None, grd_tokens=None, ut_tokens=None,
                                     use_seqscore=False, threshold=0.5,
                                     w_rel=1.0, w_sup=1.0, w_use=0.5, mode="adaptive_retrieval", closed=False):
    final_results = {}
    template_use_all_ret = prompt[0]
    template_use_one_ret = prompt[1]
    evidences_all, evidences_1 = evidences[0], evidences[1]

    # add use all evidences
    prompt_use_all_ret = [model.tokenizer.apply_chat_template(i, tokenize=False) for i in template_use_all_ret]
    prompts_use_one_ret = [model.tokenizer.apply_chat_template(i, tokenize=False) for i in template_use_one_ret]
    preds = model.generate(prompt_use_all_ret + prompts_use_one_ret)
    

    final_results["all_doc_retrieval"] = preds[0].outputs[0].text
    final_results["all_doc_retrieval_ids"] = preds[0].outputs[0].token_ids
    final_results["all_doc_retrieval_log_probs"] = preds[0].outputs[0].id_log_probs
    final_results["all_doc_retrieval_evidences"] = evidences_all
    assert len(template_use_all_ret) == 1
    final_results["all_doc_retrieval_template"] = template_use_all_ret[0]
    

    results = score_method(preds = preds[1:], rel_tokens=rel_tokens, grd_tokens=grd_tokens, ut_tokens=ut_tokens,
                                    use_seqscore=use_seqscore, w_rel=w_rel, w_sup=w_sup, w_use=w_use, prompts=template_use_one_ret, evidences=evidences_1)



    answer2score = {}
    if closed is True:
        raise NotImplementedError
        for key, result in results.items():
            if key == "no_retrieval":
                continue
            answer = postprocess_answer_option_conditioned(result["pred"])
            score = result["score"]
            answer2score.setdefault(answer, 0)
            answer2score[answer] += score
        sorted_answers = sorted(
            answer2score.items(), key=lambda x: x[1], reverse=True)
        best_option = sorted_answers[0][0]
        token_ids = []
        id_log_probs = []
    else:
        path2score = {key: item["score"] for key,
                        item in results.items() if key != "no_retrieval"}
        best_path = sorted(path2score.items(),
                            key=lambda x: x[1], reverse=True)[0][0]
        best_option = results[best_path]["pred"]
        token_ids = results[best_path]["token_ids"]
        id_log_probs = results[best_path]["id_log_probs"]
        best_evidence = results[best_path]["evidences"]
        best_template = results[best_path]["template"]
        #best_option = postprocess_answer_option_conditioned(best_option)
    final_results["retrieval"] = best_option
    final_results["retrieval_token_ids"] = token_ids
    final_results["retrieval_log_probs"] = id_log_probs
    final_results["retrieval_res"] = results
    final_results["retrieval_evidences"] = best_evidence
    final_results["retrieval_template"] = best_template
    return final_results

def process_data_evidences(demonstration, top_n):
    ctx_key = "ctxs" if "ctxs" in demonstration else "top_contexts"
    prompt = PROMPT_DICT["prompt_no_input"].format_map(demonstration)
    evidences = demonstration[ctx_key][:top_n]
    return prompt, evidences

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
        else:
            item["instruction"] = item["question"]
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
    parser.add_argument("--max_turn", type=int, default=3,
                        help="max number of turns for multi-turn conversation")
    parser.add_argument("--world_size",  type=int, default=1,
                        help="world size to use multiple GPUs.")
    parser.add_argument("--dtype",  type=str, default="half",
                        help="We use bfloat16 for training. If you run inference on GPUs that do not support BF16, please set this to be `half`.")
    # Decoding hyperparams
    parser.add_argument('--threshold', type=float,
                        default=None, help="Adaptive threshold.")
    
    parser.add_argument('--ret_model_path', type=str,
                        default=None, help="retriever model path.")
    
    parser.add_argument('--passage_path', type=str,
                        default=None, help="passage path.")
    
    parser.add_argument('--embedding_path', type=str,
                        default=None, help="embedding path.")
    
    parser.add_argument("--use_seqscore", action="store_true")
    parser.add_argument("--use_groundness", action="store_true",
                        help="use ground score")
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
    args = parser.parse_args()
    gpt = args.model_name
    input_path = args.input_file
    if input_path.endswith(".json"):
        input_data = json.load(open(input_path))
    else:
        input_data = load_jsonlines(input_path)

    input_data = preprocess_input_data(input_data, task=args.task)
    if args.task in TASK_INST:
        instruction = TASK_INST[args.task]
    else:
        instruction = None
        
    model_ = AutoModelForCausalLM.from_pretrained(gpt, device_map='auto', torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(gpt, padding_side="left")
    max_new_tokens = args.max_new_tokens
    model = MyModel(model_, tokenizer, max_new_tokens=max_new_tokens)
    
    
    

    retriever = Retriever({})
    ret_model_path = f"{MODEL_ROOT}/contriever-msmarco" if args.ret_model_path is None else args.ret_model_path
    passage_path = f"{WORK_ROOT}/retrieve_data/psgs_w100.tsv" if args.passage_path is None else args.passage_path
    embedding_path = f"{WORK_ROOT}/retrieve_data/wikipedia_embeddings/*" if args.embedding_path is None else args.embedding_path
    retriever.setup_retriever_demo(ret_model_path, passage_path, embedding_path,  n_docs=args.ndocs, save_or_load_index=True)
    
    
    score_method = sequence_scoring
    # Get token ids for reflection tokens.
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=args.use_groundness, use_utility=args.use_utility)

    def generate(prompt, evidences):
        return call_model_rerank_w_scores_batch(prompt, evidences=evidences, model=model, score_method=score_method, 
                                                rel_tokens=rel_tokens, ret_tokens=ret_tokens, grd_tokens=grd_tokens, ut_tokens=ut_tokens,
                                                threshold=args.threshold, use_seqscore=args.use_seqscore,
                                                w_rel=args.w_rel, w_sup=args.w_sup, w_use=args.w_use, mode=args.mode, closed=args.task in ["fever", "arc_c"])

    
    all_results = []
    final_results = {}
    final_results["dataset"] = args.input_file
    match_score = []
    for i, row in tqdm(enumerate(input_data)):
        res_one_data = {}
        retrieval_self_rag_prompts = []
        retrieval_self_rag_prompt = selfrag_format_prompt(prompt = row["instruction"], evidences=None, instruction=instruction)
        all_doc_retrieval_self_rag_prompts = []
        all_doc_retrieval_self_rag_prompt = selfrag_format_prompt(prompt = row["instruction"], evidences=None, instruction=instruction)
        turn = 1
        _, evidences = process_data_evidences(row, top_n=args.ndocs)
        evidences_1 = evidences
        evidences_all = evidences
        prompt = row["instruction"]
        template_1 = []
        template_all = []
        a_1 = prompt
        a_all = prompt
        while turn < args.max_turn:
            """
                def llama_template(prompt, evidences=None, instruction=None, prev_chat=[], use_all_evidences=False):

                prompt (str): the question, or previous answer
                evidences (List(Dict)): a list of evidence paragraphs
                instruction (str, optional): task instruction. Only used in the first turn. Defaults to None.
                prev_chat (List(Dict)): the best chat history. Defaults to [].
                use_all_evidences (bool, optional): whether to use all evidences. Defaults to False.
        
            """
            ret_all_prompts = llama_template(prompt=a_all, evidences=evidences_all, instruction=instruction, prev_chat=template_all, use_all_evidences=True)
            ret_prompts = llama_template(prompt=a_1, evidences=evidences_1, instruction=instruction, prev_chat=template_1, use_all_evidences=False)
            res = generate([ret_all_prompts, ret_prompts], [evidences_all, evidences_1])

            a_1 = res["retrieval"]
            a_all = res["all_doc_retrieval"]
            template_1 = res["retrieval_template"]
            template_all = res["all_doc_retrieval_template"]
            
            evidences = retriever.search_document_demo([a_1, a_all], args.ndocs)
            assert len(evidences) == 2
            evidences_1 = evidences[0]
            evidences_all = evidences[1]
            res_one_data[f"turn_{turn}"] = res
            
            retrieval_self_rag_prompt = selfrag_format_prompt(prompt = a_1, evidences=evidences_1, instruction=instruction, prev_prompt=retrieval_self_rag_prompt)
            all_doc_retrieval_self_rag_prompt = selfrag_format_prompt(prompt = a_all, evidences=evidences_all, instruction=instruction, prev_prompt=all_doc_retrieval_self_rag_prompt)
            retrieval_self_rag_prompts.append(retrieval_self_rag_prompt)
            all_doc_retrieval_self_rag_prompts.append(all_doc_retrieval_self_rag_prompt)
            turn += 1
            
            
        if 'id' in row:
            res_one_data['question_id'] = row['id']
        else:
            res_one_data['question_id'] = i # for pub health
    
        if "answers" not in row and "answer" in row:
            row["answers"] = [row["answer"]] if type(
                row["answer"]) is str else row["answer"]
        
        res_one_data['gold'] = row["answers"]
        
        res_one_data['retrieval_self_rag_prompts'] = retrieval_self_rag_prompts
        res_one_data['all_doc_retrieval_self_rag_prompts'] = all_doc_retrieval_self_rag_prompts
        
        all_results.append(res_one_data)
        
        match_score.append(match(a_1, res_one_data['gold']))
        if i % 10 == 0:
            if len(match_score) > 0:
                print("match: {} (hit max turn)".format(np.mean(match_score)))
        
    final_results["results"] = all_results
    if len(match_score) > 0:
        print("match: {}".format(np.mean(match_score)))
    with open(args.output_file, "w") as outfile:
        json.dump(final_results, outfile)

    
if __name__ == "__main__":
    main()