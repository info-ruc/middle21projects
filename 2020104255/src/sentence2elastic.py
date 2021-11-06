import json

from elasticsearch import Elasticsearch
import os
import os.path as osp
from datetime import datetime
from tqdm.contrib import tzip
import urllib.request


def index_create():
    map = {
        "mappings": {
            "properties": {
                "sentence": {
                    "type": "text",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                        }
                    }
                },
                "timestamp": {
                    "type": "date",
                },
                "features": {
                    "type": "dense_vector",
                    "dims": 512
                }
            }
        }
    }
    s = json.dumps(map)
    req = urllib.request.Request("http://127.0.0.1:9200/sentence_features",
                           data=bytes(s.encode()), method="PUT")
    req.add_header("Content-Type","application/json")
    res = urllib.request.urlopen(req)
    print(res.read().decode('utf-8'))

if __name__ == "__main__":
    # index_create()

    es = Elasticsearch()

    sentences = open("../extra/sentences.txt", "r").readlines()[12000:]
    features = open("../extra/sentence_features_backup.txt", "r").readlines()[12000:]
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
    # except Exception as e:
    #     print(e)
