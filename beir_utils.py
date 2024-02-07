import logging
from typing import List, Dict, Union, Tuple
import json
def recall_compare(qrels: Dict[str, Dict[str, int]], 
        results_origin: Dict[str, Dict[str, float]],
        results_rewrite: Dict[str, Dict[str, float]], 
        k_values: List[int]):
    
    
    better_id, worse_id = [], []
    k_max, top_hits_origin, top_hits_rewrite = max(k_values), {}, {}
    logging.info("\n")
    

    for query_id, doc_scores in results_origin.items():
        top_hits_origin[query_id] = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]   
    for query_id, doc_scores in results_rewrite.items():
        top_hits_rewrite[query_id] = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]  


    for query_id in top_hits_origin:
        query_relevant_docs = set([doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0])    
        for k in k_values:
            retrieved_docs_origin = [row[0] for row in top_hits_origin[query_id][0:k] if qrels[query_id].get(row[0], 0) > 0]
            recall_origin = len(retrieved_docs_origin) / len(query_relevant_docs)

            retrieved_docs_rewrite = [row[0] for row in top_hits_rewrite[query_id][0:k] if qrels[query_id].get(row[0], 0) > 0]
            recall_rewrite = len(retrieved_docs_rewrite) / len(query_relevant_docs)

            if recall_rewrite > recall_origin:
                better_id.append(query_id)
                break
            elif recall_rewrite < recall_origin:
                worse_id.append(query_id)
                break

    return better_id, worse_id

def precision_compare(qrels: Dict[str, Dict[str, int]], 
        results_origin: Dict[str, Dict[str, float]],
        results_rewrite: Dict[str, Dict[str, float]], 
        k_values: List[int]):
    
    
    better_id, worse_id = [], []
    k_max, top_hits_origin, top_hits_rewrite = max(k_values), {}, {}
    logging.info("\n")
    

    for query_id, doc_scores in results_origin.items():
        top_hits_origin[query_id] = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]   
    for query_id, doc_scores in results_rewrite.items():
        top_hits_rewrite[query_id] = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]  


    for query_id in top_hits_origin: 
        for k in k_values:
            retrieved_docs_origin = [row[0] for row in top_hits_origin[query_id][0:k] if qrels[query_id].get(row[0], 0) > 0]
            precision_origin = len(retrieved_docs_origin) / k

            retrieved_docs_rewrite = [row[0] for row in top_hits_rewrite[query_id][0:k] if qrels[query_id].get(row[0], 0) > 0]
            precision_rewrite = len(retrieved_docs_rewrite) / k

            if precision_rewrite > precision_origin:
                better_id.append(query_id)
                break
            elif precision_rewrite < precision_origin:
                worse_id.append(query_id)
                break

    return better_id, worse_id

def f1_compare(qrels: Dict[str, Dict[str, int]], 
        results_origin: Dict[str, Dict[str, float]],
        results_rewrite: Dict[str, Dict[str, float]], 
        k_values: List[int]):
    
    
    better_id, worse_id = [], []
    k_max, top_hits_origin, top_hits_rewrite = max(k_values), {}, {}
    logging.info("\n")
    

    for query_id, doc_scores in results_origin.items():
        top_hits_origin[query_id] = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]   
    for query_id, doc_scores in results_rewrite.items():
        top_hits_rewrite[query_id] = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]  


    for query_id in top_hits_origin:
        query_relevant_docs = set([doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0])    
        for k in k_values:
            retrieved_docs_origin = [row[0] for row in top_hits_origin[query_id][0:k] if qrels[query_id].get(row[0], 0) > 0]
            recall_origin = len(retrieved_docs_origin) / len(query_relevant_docs)
            precision_origin = len(retrieved_docs_origin) / k
            if (precision_origin + recall_origin) == 0:
                f1_origin = 0
            else:
                f1_origin = 2 * precision_origin * recall_origin / (precision_origin + recall_origin)

            retrieved_docs_rewrite = [row[0] for row in top_hits_rewrite[query_id][0:k] if qrels[query_id].get(row[0], 0) > 0]
            recall_rewrite = len(retrieved_docs_rewrite) / len(query_relevant_docs)
            precision_rewrite = len(retrieved_docs_rewrite) / k
            if (precision_rewrite + recall_rewrite) == 0:
                f1_rewrite = 0
            else:
                f1_rewrite = 2 * precision_rewrite * recall_rewrite / (precision_rewrite + recall_rewrite)

            if f1_rewrite > f1_origin:
                better_id.append(query_id)
                break
            elif f1_rewrite < f1_origin:
                worse_id.append(query_id)
                break

    return better_id, worse_id

def mrr_compare(qrels: Dict[str, Dict[str, int]], 
        results_origin: Dict[str, Dict[str, float]],
        results_rewrite: Dict[str, Dict[str, float]], 
        k_values: List[int]):
    
    better_id, worse_id = [], []
    k_max, top_hits_origin, top_hits_rewrite = max(k_values), {}, {}
    logging.info("\n")
    
    for query_id, doc_scores in results_origin.items():
        top_hits_origin[query_id] = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]   
    for query_id, doc_scores in results_rewrite.items():
        top_hits_rewrite[query_id] = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]  


    for query_id in top_hits_origin:
        query_relevant_docs = set([doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0])    
        for k in k_values:
            mrr_origin, mrr_rewrite = 0, 0
            for rank, hit in enumerate(top_hits_origin[query_id][0:k]):
                if hit[0] in query_relevant_docs:
                    mrr_origin = 1.0 / (rank + 1)
                    break
            for rank, hit in enumerate(top_hits_rewrite[query_id][0:k]):
                if hit[0] in query_relevant_docs:
                    mrr_rewrite = 1.0 / (rank + 1)
                    break
            if mrr_rewrite > mrr_origin:
                better_id.append(query_id)
                break
            elif mrr_rewrite < mrr_origin:
                worse_id.append(query_id)
                break

    return better_id, worse_id

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

