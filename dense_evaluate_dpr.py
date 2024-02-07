from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib, os
import random

from typing import List, Dict, Union, Tuple

def retrieval(data_path: str) -> Dict[str, Dict[str, float]]:

    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])


    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="dev")


    model = DRES(models.SentenceBERT((
        "models/facebook-dpr-question_encoder-multiset-base",
        "models/facebook-dpr-ctx_encoder-multiset-base",
        " [SEP] "), batch_size=128))




    retriever = EvaluateRetrieval(model, score_function="dot")

    results = retriever.retrieve(corpus, queries)


    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

    top_k = 10

    query_id, ranking_scores = random.choice(list(results.items()))
    scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
    logging.info("Query : %s\n" % queries[query_id])


if __name__ == '__main__':

    data_path1 = 'datasets/trec-covid'
    data_path2 = 'datasets/trec-covid/trec-covid_new'
    k_values = [1,3,5,10,100,1000]

    retrieval(data_path=data_path1)
    write_results(data_path=data_path1, results=results_origin, pattern='origin_sbert')

    
    retrieval(data_path=data_path2)