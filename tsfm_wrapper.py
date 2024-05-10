from transformers import AutoModelForCausalLM, GenerationConfig, AutoTokenizer
from transformers.generation.utils import GreedySearchDecoderOnlyOutput, BeamSampleDecoderOnlyOutput


class OutputRequest:
    def __init__(self, output_unit):
        self.outputs = [output_unit]

class OutputRequest_unit:
    def __init__(self, token_ids, text, logprobs, cumulative_logprob, id_log_probs):
        self.token_ids = token_ids
        self.text = text
        self.logprobs = logprobs
        self.cumulative_logprob = cumulative_logprob
        self.id_log_probs = id_log_probs

class MyModel:
    def __init__(self, model, tokenizer, max_new_tokens=100):
        print("#====================================================#")
        print(f"# Configuring model Wrapper with max_new_tokens={max_new_tokens}")
        self.model = model
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            special_tokens_dict = {'pad_token': '[PAD]'}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.model.resize_token_embeddings(len(self.tokenizer))
            print(f"# LM tokenizer does not contain padding token, set pad_token to {self.tokenizer.pad_token}: {self.tokenizer.pad_token_id}")
        self.pad_token = self.tokenizer.pad_token
        self.pad_token_id = self.tokenizer.pad_token_id

        try:
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        except:
            terminators = [
                tokenizer.eos_token_id
            ]
        print(f"# Terminators: {' '.join([f'{i}: {tokenizer.decode(i)}' for i in terminators])}")
        print("#====================================================#")
        # greedy decoding
        self.generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens, renormalize_logits=True, return_dict_in_generate=True, output_scores=True, temperature=None, pad_token_id=self.pad_token_id, eos_token_id=terminators) 
        self.gen_config_dc = {'max_new_tokens': max_new_tokens, 'renormalize_logits': True, 'return_dict_in_generate': True, 'output_scores': True, 'temperature': None, 'pad_token_id': self.pad_token_id, 'eos_token_id': terminators}
    def generate(self, queries, **kwargs):
        if not kwargs:
            generation_config = self.generation_config
        else:
            gen_config_dc = self.gen_config_dc.copy()
            gen_config_dc.update(kwargs)
            generation_config = GenerationConfig(**gen_config_dc)
        
        batch_size = len(queries)
        inputs = self.tokenizer(queries, padding=True, return_tensors="pt").to(self.model.device)
        start_ind = inputs['input_ids'].shape[1]   
        tsfm_outputs = self.model.generate(**inputs, generation_config=generation_config)
        
        token_ids = tsfm_outputs.sequences[..., start_ind:] # token_ids starts from the query, and are padded to the same length
        texts = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        print(texts)
        print(token_ids.shape)
        token_ids = token_ids.tolist()

        log_probs = tsfm_outputs.scores # a tuple of size max output length, each element is a tensor of size batch_size(*beam size) x vocab_size [L, B*b, vocab]
        # score: https://huggingface.co/docs/transformers/v4.37.2/en/internal/generation_utils#transformers.LogitNormalization.example
        print(log_probs.shape)
        log_probs_dict = [[] for _ in range(batch_size * generation_config.num_beams * generation_config.num_return_sequences)]
        id_log_probs = [[] for _ in range(batch_size * generation_config.num_beams * generation_config.num_return_sequences)]
        if isinstance(tsfm_outputs, BeamSampleDecoderOnlyOutput):
            cumul_log_probs = tsfm_outputs.sequences_scores
        else:
            cumul_log_probs = [0.0 for _ in range(batch_size)] # set default value for now, TBC

        for pos, element in enumerate(log_probs): 
            element = element.tolist()
            for i, score in enumerate(element): # iterate over batch_size(*beam size)
                id2prob = {token_id: p for token_id, p in enumerate(score)}
                log_probs_dict[i].append(id2prob)
                if isinstance(tsfm_outputs, GreedySearchDecoderOnlyOutput) and token_ids[i][pos] != self.pad_token_id:
                    cumul_log_probs[i] += id2prob[token_ids[i][pos]] 
                    id_log_probs[i].append(id2prob[token_ids[i][pos]])
        preds = []
        assert len(token_ids) == len(texts) == len(log_probs_dict) == len(cumul_log_probs)
        assert len(token_ids[0]) == len(log_probs_dict[0])
        assert len(log_probs_dict[0][0]) == len(self.tokenizer)
        for i in zip (token_ids, texts, log_probs_dict, cumul_log_probs, id_log_probs):
            preds.append(OutputRequest(OutputRequest_unit(*i)))
        
        return preds


if __name__ == '__main__':
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
    MODEL_ROOT = os.environ.get("model_path")
    import torch
    path = f"{MODEL_ROOT}/Llama-2-7b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(path, device_map='auto', torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
    max_new_tokens = 10
    model = MyModel(model, tokenizer, max_new_tokens=max_new_tokens)
    queries = ["An increasing sequence: one,", "how are you?", "Tell me 1+1=? Answer a number:"]

    import time
    start_time = time.time()
    preds = model.generate(queries, num_return_sequences=3, do_sample=True, temperature=1.0)
    print(f"Generation takes time {time.time()-start_time}")
    for i in preds:
        print(i.outputs[0].text)
        print(i.outputs[0].token_ids)
        for j, token_id in enumerate(i.outputs[0].token_ids):
            print(token_id, "|", tokenizer.decode(token_id), "|", i.outputs[0].id_log_probs[j])
        print(i.outputs[0].id_log_probs)