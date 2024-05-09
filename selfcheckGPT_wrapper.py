from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Tuple, Any, Union
import torch
from selfcheckgpt.modeling_selfcheck import SelfCheckNLI

class MyNLI:
    def __init__(self, model_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.selfcheck_nli = SelfCheckNLI(nli_model=model_path,device=device) 
        

    # def predict(self, model, queries: List[str], passages: List[str]):
    #     inputs = self.tokenizer.batch_encode_plus(
    #         batch_text_or_text_pairs=[(sent, passage) for sent in sentences for passage in sampled_passages],
    #         add_special_tokens=True, return_tensors="pt",
    #     )
    #     logits = self.model(**inputs).logits
    #     probs = torch.softmax(logits, dim=-1)[:, 0]
    #     sent_scores = probs.view(len(sentences), len(sampled_passages))
    #     return sent_scores
    
    def compute_nli_score(self, examples: Union[List[Tuple[str, str]], Tuple[str, str]]):
        if isinstance(examples, tuple):  
            examples = [examples]
        inputs = self.selfcheck_nli.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=examples,
            add_special_tokens=False, return_tensors="pt",
        ).to(self.model.device)
        logits = self.selfcheck_nli.model(**inputs).logits # neutral is already removed
        probs = torch.softmax(logits, dim=-1)
        
        return probs
    
    
# textA = "Kyle Walker has a personal issue"
# textB = "Kyle Walker will remain Manchester City captain following reports about his private life, says boss Pep Guardiola."

# inputs = tokenizer.batch_encode_plus(
#     batch_text_or_text_pairs=[(textA, textB)],
#     add_special_tokens=True, return_tensors="pt",
# )
# logits = model(**inputs).logits # neutral is already removed
# probs = torch.softmax(logits, dim=-1)[0]
# # probs = [0.7080, 0.2920], meaning that prob(entail) = 0.708, prob(contradict) = 0.292

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
    MODEL_ROOT = os.environ.get("model_path")
    
    model_path = f"{MODEL_ROOT}/deberta-v3-large-mnli"
    selfcheck_nli = MyNLI(model_path)
    examples = [
        ("Kyle Walker has a personal issue", "Kyle Walker will remain Manchester City captain following reports about his private life, says boss Pep Guardiola."),
        ("I like you.", "I love you.")
    ]
    probs = selfcheck_nli.compute_nli_score(examples)
    print(probs)
    # tensor([[0.7080, 0.2920],
    #         [0.7080, 0.2920],
    #         [0.7080, 0.2920]], grad_fn=<SoftmaxBackward>)