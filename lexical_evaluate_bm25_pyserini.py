
from beir_utils import mrr_compare, recall_compare, write_results, read_results, write_compare_results
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

import pathlib, os, json
import logging
import requests
import random
from typing import List, Dict, Union, Tuple

import time
def retrieval(data_path: str) -> Dict[str, Dict[str, float]]:
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])


    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

    pyserini_jsonl = "pyserini.jsonl"
    with open(os.path.join(data_path, pyserini_jsonl), 'w', encoding="utf-8") as fOut:
        for doc_id in corpus:
            title, text = corpus[doc_id].get("title", ""), corpus[doc_id].get("text", "")
            data = {"id": doc_id, "title": title, "contents": text}
            json.dump(data, fOut)
            fOut.write('\n')

    docker_beir_pyserini = "http://127.0.0.1:8000"

    with open(os.path.join(data_path, "pyserini.jsonl"), "rb") as fIn:
        r = requests.post(docker_beir_pyserini + "/upload/", files={"file": fIn}, verify=False)

    index_name = "beir/scifact" # beir/scifact
    r = requests.get(docker_beir_pyserini + "/index/", params={"index_name": index_name})

    retriever = EvaluateRetrieval()
    qids = list(queries)
    query_texts = [queries[qid] for qid in qids]
    payload = {"queries": query_texts, "qids": qids, "k": max(retriever.k_values)}

    time1 = time.time()
    results = json.loads(requests.post(docker_beir_pyserini + "/lexical/batch_search/", json=payload).text)["results"]
    print(time.time() - time1)
    for query_id in results:
        if query_id in results[query_id]:
            results[query_id].pop(query_id, None)

    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

    query_id, scores_dict = random.choice(list(results.items()))
    logging.info("Query : %s\n" % queries[query_id])

    scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
    return results, qrels
if __name__ == '__main__':

    data_path1 = 'datasets/scifact'
    data_path2 = 'datasets/scifact/scifact_new'
    

    #results_origin, qrels = retrieval(data_path=data_path1)
    #write_results(data_path=data_path1, results=results_origin, pattern='origin_bm25_train')

    results_rewrite, qrels = retrieval(data_path=data_path2)
    #write_results(data_path=data_path2, results=results_rewrite, pattern='rewrite_bm25_s2_specialz_dev')
    
    #corpus, queries, qrels = GenericDataLoader(data_path1).load(split="dev")
    #results_origin = read_results(data_path=data_path1, pattern='origin_bm25_dev')
    #results_rewrite = read_results(data_path=data_path2, pattern='rewrite_bm25_t1_train')

    #k_values = [10, 20, 100, 1000]
    #better_id, worse_id = mrr_compare(qrels=qrels, results_origin=results_origin, results_rewrite=results_rewrite, k_values=k_values)
    #write_compare_results(data_path=data_path1, better_id=better_id, worse_id=worse_id, pattern='mrr_bm25_s2_specialz_temp3')
    #print(len(better_id), len(worse_id))

    #better_id, worse_id = recall_compare(qrels=qrels, results_origin=results_origin, results_rewrite=results_rewrite, k_values=k_values)
    #write_compare_results(data_path=data_path1, better_id=better_id, worse_id=worse_id, pattern='recall_bm25_s2_specialz_temp3')  
    #print(len(better_id), len(worse_id))
