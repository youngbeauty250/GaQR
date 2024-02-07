from time import time
from beir_utils import mrr_compare, recall_compare
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib, os, json
import random
from typing import List, Dict, Union, Tuple

def retrieval(data_path: str) -> Dict[str, Dict[str, float]]:
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    model = models.SentenceBERT("models/msmarco-distilbert-base-tas-b")
    model = DRES(model, batch_size=256, corpus_chunk_size=512*9999)
    retriever = EvaluateRetrieval(model, score_function="dot")
    start_time = time()
    results = retriever.retrieve(corpus, queries)
    end_time = time()
    print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))
    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

    mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
    recall_cap = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="r_cap")
    hole = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="hole")

    top_k = 1

    query_id, ranking_scores = random.choice(list(results.items()))
    scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
    logging.info("Query : %s\n" % queries[query_id])

    return results, qrels

def write_results(data_path: str, results: Dict[str, Dict[str, float]], pattern: str):
    with open(data_path + '/{}_results.jsonl'.format(pattern), "w") as write_file:
        for query_id, doc_scores in results.items():
            js = {}
            js[query_id] = doc_scores
            json.dump(js, write_file, ensure_ascii=False)
            write_file.write("\n")

def read_results(data_path: str,  pattern: str) -> Dict[str, Dict[str, float]]:
    with open(data_path + '/{}_results.jsonl'.format(pattern), "r") as write_file:
        results = {}
        for js in write_file:
            js = json.loads(js)
            for query_id, doc_scores in js.items():
                results[query_id] = doc_scores
        return results

def write_compare_results(data_path: str, better_id: List[str], worse_id: List[str], pattern: str):
    with open(data_path + '/{}_compare_results.jsonl'.format(pattern), "w") as write_file:
        js = {}
        js['better_id'] = better_id
        js['worse_id'] = worse_id
        json.dump(js, write_file, ensure_ascii=False)
        write_file.write("\n")


if __name__ == '__main__':

    data_path1 = 'datasets/trec-covid'
    data_path2 = 'datasets/trec-covid/trec-covid_new'
    k_values = [1,3,5,10,100,1000]

    results_origin, qrels = retrieval(data_path=data_path1)
    #write_results(data_path=data_path1, results=results_origin, pattern='origin_sbert')

    
    results_rewrite, qrels = retrieval(data_path=data_path2)
    #write_results(data_path=data_path2, results=results_rewrite, pattern='rewrite_sbert')

