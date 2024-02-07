import requests
import json
from flask import Flask, request, jsonify
from tqdm import tqdm
import csv
app = Flask(__name__)

def query_LLM(model_url, query):
    answer = requests.post(
        model_url,
        json={'prompt': query}
    ).json()
    return answer['response']

def write_qrels(qrels_file, write_file):
    with open(qrels_file, "r") as f:
        qrels = f.readlines()
        head = qrels[0]
        qrels = qrels[30001:40001]
    with open(write_file, "w") as f:
        f.write(head)
        f.writelines(qrels)
    
def read_qrels(qrels_file):
    qrels = {}
    reader = csv.reader(open(qrels_file, encoding="utf-8"), 
                            delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    next(reader)
    
    for id, row in enumerate(reader):
        query_id, corpus_id, score = row[0], row[1], int(row[2])
        
        if query_id not in qrels:
            qrels[query_id] = {corpus_id: score}
        else:
            qrels[query_id][corpus_id] = score
    return qrels


def continue_test():
    num_epoch = 4
    train_tsv = "/beir/datasets/hotpotqa/qrels/train.tsv"
    temp_tsv = "/beir/datasets/hotpotqa/qrels/train_temp{}.tsv".format(num_epoch)
    temp_tsv_new = "/beir/datasets/hotpotqa/hotpotqa_new/qrels/train_temp{}.tsv".format(num_epoch)
    write_qrels(train_tsv, temp_tsv)
    write_qrels(train_tsv, temp_tsv_new)
    qrels = read_qrels(temp_tsv)
    with open("/beir/datasets/hotpotqa/queries.jsonl", "r", encoding="utf-8") as read_file:
        with open("data/test_data/hotpotqa_temp{}_s2_specialz_queries_results.jsonl".format(num_epoch), "w", encoding="utf-8") as write_file:
            num = 0
            for data in read_file:
                line = json.loads(data)
                if line["_id"] not in qrels.keys():
                   continue
                num += 1
                if num % 10 == 0:
                    print(num)

                js = line
                js['origin_query'] = line['text']
                instruction = "You are a search engine. In order to obtain information for answering the query, please provide at least three rewritten queries. Do not answer the rewritten queries. Don't output any words other than the rewritten queries. The rewritten queries are split by '###'. Below are a query:\nquery: "
                query = instruction + line['text']

                response = query_LLM('###', query)
                js['text'] = response
                
                json.dump(js, write_file, ensure_ascii=False)
                write_file.write("\n")
    data_transfer()
def data_transfer():
    dataset = 'scifact'
    data_path2 = '/data/test_data/{}_s2_dev_queries_results.jsonl'.format(dataset)


    store_path2 = '/beir/datasets/{}/{}_new/queries3.jsonl'.format(dataset, dataset)
    with open(data_path2, "r", encoding="utf-8") as f:
        results = []
        for line in f:
            js = {}
            line = json.loads(line)
            js = line
            js['text'] = js['origin_query'] * 5 + js['text']

            results.append(js)

        with open(store_path2, "w", encoding='utf-8') as k:
            for result in results:
                json.dump(result, k, ensure_ascii=False)
                k.write("\n")
import time
def test():

    dataset = 'scifact'
    pattern = 'test'
    time_start = time.time() # 记录开始时间
    qrels = read_qrels("/beir/datasets/{}/qrels/{}.tsv".format(dataset, pattern))
    with open("/beir/datasets/{}/queries.jsonl".format(dataset), "r", encoding="utf-8") as read_file:
        with open("data/test_data/{}_s2_dev_queries_results.jsonl".format(dataset), "w", encoding="utf-8") as write_file:
            num = 0
            for data in read_file:
                line = json.loads(data)
                if line["_id"] not in qrels.keys():
                   continue
                num += 1
                if num % 10 == 0:
                    print(num)

                js = line
                js['origin_query'] = line['text']
                #instruction = "You are a search engine. In order to obtain information for answering the query, please provide at least three rewritten queries. Do not answer the rewritten queries. Don't output any words other than the rewritten queries. The rewritten queries are split by '###'. Below are a query:\nquery: "
                instruction = "Answer the following query: {} \nGive the rationale before answering"
                query = instruction.format(line['text'])

                response = query_LLM('###', query)
                js['text'] = response
                json.dump(js, write_file, ensure_ascii=False)
                write_file.write("\n")
    time_end = time.time() # 记录结束时间
    time_sum = time_end - time_start
    print(time_sum)
test()
data_transfer()