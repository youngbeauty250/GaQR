from beir_utils import mrr_compare, recall_compare, precision_compare, f1_compare
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25

import pathlib, os, random, json
import logging
from typing import List, Dict, Union, Tuple


def retrieval(data_path: str) -> Dict[str, Dict[str, float]]:
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])


    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")


    hostname = "localhost:9200/" #localhost
    index_name = "trec-covid" # trec-covid

    initialize = True # False


    model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
    retriever = EvaluateRetrieval(model)

    results = retriever.retrieve(corpus, queries)

    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

    query_id, scores_dict = random.choice(list(results.items()))
    logging.info("Query : %s\n" % queries[query_id])

    scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
    for rank in range(10):
        doc_id = scores[rank][0]
        logging.info("Doc %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))
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
    

    results_origin, qrels = retrieval(data_path=data_path1)
    #write_results(data_path=data_path1, results=results_origin, pattern='origin_bm25_train')

    results_rewrite, qrels = retrieval(data_path=data_path2)
    
    k_values = [10, 20, 100, 1000]
    better_id, worse_id = mrr_compare(qrels=qrels, results_origin=results_origin, results_rewrite=results_rewrite, k_values=k_values)
    print(len(better_id), len(worse_id))

    better_id, worse_id = recall_compare(qrels=qrels, results_origin=results_origin, results_rewrite=results_rewrite, k_values=k_values)
    print(len(better_id), len(worse_id))
