from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Tuple, Any, Union
import torch
from selfcheckgpt.modeling_selfcheck import SelfCheckNLI

class MyNLI:
    def __init__(self, model_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if 'deberta-v3-large-mnli' in model_path: 
            selfcheck_nli = SelfCheckNLI(nli_model=model_path,device=device) 
            self.model = selfcheck_nli.model
            self.tokenizer = selfcheck_nli.tokenizer
        else:   
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model.eval()
            self.model.to(device)
                    

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
        inputs = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=examples,
            add_special_tokens=True, padding="longest",
            truncation=True, return_tensors="pt",
        ).to(self.model.device)
        logits = self.model(**inputs).logits # neutral is already removed
        probs = torch.softmax(logits, dim=-1)
        
        return probs
    
    

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
    for i, prob in enumerate(probs):
        print("#====================================================#")
        print(f"# Premise: {examples[i][0]}")
        print(f"# Hypothesis: {examples[i][1]}")
        print(f"# EntailmentP: {prob[0]}")
        print(f"# ContradictP: {prob[1]}")
    print("#====================================================#")
    # tensor([[0.708, 0.292],
    #         [0.998, 0.002]], grad_fn=<SoftmaxBackward>)
    examples = ("I like you.", "I love you")
    probs = selfcheck_nli.compute_nli_score(examples)
    print(probs)