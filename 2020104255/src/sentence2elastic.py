import json
import requests
from elasticsearch import Elasticsearch
import os
import os.path as osp
from datetime import datetime
from tqdm.contrib import tzip
import urllib.request


def index_create():
    map = {
        "settings": {
            "index": {
                "number_of_shards": "2",
                "number_of_replicas": "0"
            }
        },
        "mappings": {
            "properties": {
                    "sentence": {
                        "type": "text",
                    },
                    "timestamp": {
                        "type": "date",
                    },
                    "feature": {
                        "type": "dense_vector",
                        "dims": 512
                    }
                }
        }
    }
    res = requests.put(
        url="http://127.0.0.1:9200/sentence_features/",
        data=bytes(json.dumps(map).encode()),
        headers={
            "Content-Type":"application/json"
        }
    )

    print(res.text)

if __name__ == "__main__":
    index_create()

    es = Elasticsearch()

    sentences = open("../extra/sentences.txt", "r").readlines()
    features = open("../extra/sentence_features_backup.txt", "r").readlines()
    index = "sentence_features"
    for sentence, feature_str in tzip(sentences, features):
        try:
            feature = list(map(float, feature_str.split(" ")))
        except:
            print("-------")
            continue


        body = {
            "sentence": sentence,
            "feature": feature,
            "timestamp": datetime.now()
        }
        # try:
        es.index(index=index, document=body, refresh=False)
        # break
    # except Exception as e:
    #     print(e)
