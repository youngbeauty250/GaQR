from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.sparse import SparseSearch
from beir_utils import mrr_compare, recall_compare, precision_compare, f1_compare, write_compare_results, read_results, write_results

import logging
import pathlib, os
import random
from typing import List, Dict, Union, Tuple

def retrieval(data_path: str) -> Dict[str, Dict[str, float]]:
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])


    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")


    model_path = "models/sparta-msmarco-distilbert-base-v1"
    sparse_model = SparseSearch(models.SPARTA(model_path), batch_size=128)
    retriever = EvaluateRetrieval(sparse_model)

    results = retriever.retrieve(corpus, queries)


    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

    top_k = 1

    query_id, ranking_scores = random.choice(list(results.items()))
    scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
    logging.info("Query : %s\n" % queries[query_id])

    return results, qrels

if __name__ == '__main__':

    data_path1 = 'datasets/trec-covid'
    data_path2 = 'datasets/trec-covid/trec-covid_new'
    k_values = [1,3,5,10,100,1000]

    results_origin, qrels = retrieval(data_path=data_path1)
    #write_results(data_path=data_path1, results=results_origin, pattern='origin_sparta')

    results_rewrite, qrels = retrieval(data_path=data_path2)
    #write_results(data_path=data_path2, results=results_rewrite, pattern='rewrite_sparta')
    
    #corpus, queries, qrels = GenericDataLoader(data_path1).load(split="test")
    #results_origin = read_results(data_path=data_path1, pattern='origin_sparta')
    #results_rewrite = read_results(data_path=data_path2, pattern='rewrite_sparta')

    
    #better_id, worse_id = mrr_compare(qrels=qrels, results_origin=results_origin, results_rewrite=results_rewrite, k_values=k_values)
    #write_compare_results(data_path=data_path1, better_id=better_id, worse_id=worse_id, pattern='mrr_sparta')
    #print(len(better_id), len(worse_id))

    #better_id, worse_id = recall_compare(qrels=qrels, results_origin=results_origin, results_rewrite=results_rewrite, k_values=k_values)
    #write_compare_results(data_path=data_path1, better_id=better_id, worse_id=worse_id, pattern='recall_sparta')  
    #print(len(better_id), len(worse_id))