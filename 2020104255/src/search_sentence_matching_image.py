import numpy as np
import torch
import os
import os.path as osp
from tqdm import tqdm
from elasticsearch import Elasticsearch
import clip
from PIL import Image

if __name__ == "__main__":
    image_path = "../extra/example/cat.jpeg"
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
    res = es.search(body=body,index="sentence_features",timeout='100s')
    hits = res["hits"]["hits"]
    print(res)
