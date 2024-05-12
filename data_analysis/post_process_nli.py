from utils import *
import os
from selfcheckGPT_wrapper import MyNLI
import numpy as np
import json
from dotenv import load_dotenv, find_dotenv
from functools import singledispatch

load_dotenv(find_dotenv())
MODEL_ROOT = os.environ.get("model_path")

model_path = f"{MODEL_ROOT}/deberta-v3-large-mnli"
selfcheck_nli = MyNLI(model_path)
# probs = selfcheck_nli.compute_nli_score(examples)

res = []
data = load_file("llama2chat-tqa.json")
for i, data_item_ in enumerate(data):
    data_item = data_item_.copy()
    pred_types = ["all_doc_retrieval", "no_retrieval", "ret_0", "ret_1", "ret_2", "ret_3", "ret_4"]
    q = data_item["question"]
    for pred_type in pred_types:
        a = data_item[pred_type]
        
        if pred_type != 'no_retrieval':
            c = data_item[pred_type + "_ctx"]
            nil_candidates = [(q, a), (q, c), (c, a)]
        else:
            nil_candidates = [(q, a)]
        nil_scores = selfcheck_nli.compute_nli_score(nil_candidates).cpu().detach().numpy()
        data_item[pred_type + "_scores"]['qa'] = nil_scores[0][0]
        if pred_type != 'no_retrieval':
            data_item[pred_type + "_scores"]['qc'] = nil_scores[1][0]
            data_item[pred_type + "_scores"]['ca'] = nil_scores[2][0]

    data_item["no_retrieval_scores"]['qc'] = np.mean([data_item[f"ret_{i}_scores"]['qc'] for i in range(5)] + [data_item["all_doc_retrieval_scores"]['qc']])
    data_item["no_retrieval_scores"]['ca'] = np.mean([data_item[f"ret_{i}_scores"]['ca'] for i in range(5)] + [data_item["all_doc_retrieval_scores"]['ca']])
    res.append(data_item)


@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)


@to_serializable.register(np.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return np.float64(val)
with open("llama2chat-tqa_processed.json", "w") as f:
    json.dump(res, f, default=to_serializable)