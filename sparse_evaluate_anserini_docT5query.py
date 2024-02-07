
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.generation.models import QGenModel
from tqdm.autonotebook import trange

import pathlib, os, json
import logging
import requests
import random
from typing import List, Dict, Union, Tuple

def retrieval(data_path: str) -> Dict[str, Dict[str, float]]:
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    corpus_ids = list(corpus.keys())
    corpus_list = [corpus[doc_id] for doc_id in corpus_ids]


    model_path = "models/doc2query-t5-base-msmarco"
    qgen_model = QGenModel(model_path, use_fast=False)

    gen_queries = {} 
    num_return_sequences = 3
    batch_size = 80

    for start_idx in trange(0, len(corpus_list), batch_size, desc='question-generation'):            
        
        size = len(corpus_list[start_idx:start_idx + batch_size])
        ques = qgen_model.generate(
            corpus=corpus_list[start_idx:start_idx + batch_size], 
            ques_per_passage=num_return_sequences,
            max_length=64,
            top_p=0.95,
            top_k=10)
        
        assert len(ques) == size * num_return_sequences
        
        for idx in range(size):
            start_id = idx * num_return_sequences
            end_id = start_id + num_return_sequences
            gen_queries[corpus_ids[start_idx + idx]] = ques[start_id: end_id]

    pyserini_jsonl = "pyserini.jsonl"
    with open(os.path.join(data_path, pyserini_jsonl), 'w', encoding="utf-8") as fOut:
        for doc_id in corpus:
            title, text = corpus[doc_id].get("title", ""), corpus[doc_id].get("text", "")
            query_text = " ".join(gen_queries[doc_id])
            data = {"id": doc_id, "title": title, "contents": text, "queries": query_text}
            json.dump(data, fOut)
            fOut.write('\n')

    docker_beir_pyserini = ""

    with open(os.path.join(data_path, "pyserini.jsonl"), "rb") as fIn:
        r = requests.post(docker_beir_pyserini + "/upload/", files={"file": fIn}, verify=False)


    index_name = "beir/trec-covid"
    r = requests.get(docker_beir_pyserini + "/index/", params={"index_name": index_name})



    retriever = EvaluateRetrieval()
    qids = list(queries)
    query_texts = [queries[qid] for qid in qids]
    payload = {"queries": query_texts, "qids": qids, "k": max(retriever.k_values), 
            "fields": {"contents": 1.0, "title": 1.0, "queries": 1.0}}

    results = json.loads(requests.post(docker_beir_pyserini + "/lexical/batch_search/", json=payload).text)["results"]


    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)


    query_id, scores_dict = random.choice(list(results.items()))
    logging.info("Query : %s\n" % queries[query_id])

    scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
if __name__ == '__main__':

    data_path1 = 'datasets/trec-covid'
    data_path2 = 'datasets/trec-covid/trec-covid_new'
    k_values = [1,3,5,10,100,1000]

    retrieval(data_path=data_path1)


    
    retrieval(data_path=data_path2)