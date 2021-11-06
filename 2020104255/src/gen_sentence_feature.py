import torch
import clip
from elasticsearch import Elasticsearch
from datetime import datetime
from tqdm import trange
if __name__ == "__main__":
    torch.set_num_threads(8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)
    es = Elasticsearch()
    with open("../extra/sentences.txt","r") as f:
        sentences = f.readlines()
    # for i in range(len(sentences)):
    #     sentences[i] = sentences[i] if len(sentences[i])<70 else " "
    sentence_splits = []
    batch_size = 4
    with open("../extra/sentence_features.txt","w") as f:
        for i in trange(0,len(sentences),batch_size):
            text = clip.tokenize(sentences[i:i+batch_size],truncate=True)
            with torch.no_grad():
                res = model.encode_text(text)
            res = res.numpy().tolist()
            for r in res:
                f.write(" ".join(map(str,r)) + "\n")



