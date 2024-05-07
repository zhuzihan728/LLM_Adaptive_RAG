from passage_retrieval import Retriever

import os
import torch
import src.slurm
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
MODEL_ROOT = os.environ.get("model_path")
WORK_ROOT = os.environ.get("work_path")
retriever = Retriever({})
model_path = f"{MODEL_ROOT}/contriever-msmarco"
passage_path = f"{WORK_ROOT}/retrieve_data/psgs_w100.tsv"
embedding_path = f"{WORK_ROOT}/retrieve_data/wikipedia_embeddings/*"
retriever.setup_retriever_demo(model_path, passage_path, embedding_path,  n_docs=5, save_or_load_index=True)
retrieved_documents = retriever.search_document_demo(["Can you tell me the difference between llamas and alpacas?", "What is overfitting"], 5)
print(retrieved_documents)
retrieved_documents = retriever.search_document_demo(["What is overfitting"], 5)
print(retrieved_documents)