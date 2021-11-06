import numpy as np
import torch
import os
import os.path as osp
from tqdm import tqdm
from elasticsearch import Elasticsearch
import clip
from PIL import Image

if __name__ == "__main__":
    # image_path = "/Users/yyj/PycharmProjects/middle21projects/2020104255/extra/Arxiv6K.part4/2001.00338/chem2.png"
    # image_path = "/Users/yyj/PycharmProjects/middle21projects/2020104255/extra/Arxiv6K.part4/2001.08779/chart.png"
    # image_path = "/Users/yyj/PycharmProjects/middle21projects/2020104255/extra/Arxiv6K.part4/2010.01676/CNN.png"
    image_path = "/Users/yyj/PycharmProjects/middle21projects/2020104255/extra/Arxiv6K.part4/2010.02354/images/torch_code_white.png"
    es = Elasticsearch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)
    image = Image.open(image_path).convert("RGB")
    img_convert = preprocess(image)
    img_convert = torch.tensor(np.stack([img_convert]))
    with torch.no_grad():
        image_feature = model.encode_image(img_convert)[0].numpy().tolist()
    body = {
        "size": 5,
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.queryVector, doc['feature']) + 1.0",
                    "params": {
                        "queryVector": image_feature
                    }
                }
            }
        }
    }
    res = es.search(body=body,index="sentence_features",timeout="10m")
    hits = res["hits"]["hits"]
    for hit in hits:
        print(
            f"score: {hit['_score']}, sentence: {hit['_source']['sentence'].strip()}"
        )
    # print(hits)
