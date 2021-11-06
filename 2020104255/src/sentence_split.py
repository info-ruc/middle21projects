import os
import os.path as osp
import nltk
from tqdm import tqdm

if __name__ == "__main__":
    root_path = "../extra/tex2plain/"
    save_path = "../extra/sentences.txt"
    with open(save_path, "a") as ff:
        for file_name in tqdm(os.listdir(root_path)):
            file_path = osp.join(root_path,file_name)
            with open(file_path,"r") as f:
                article = f.read().replace("\n"," ").replace("  "," ")
                res = nltk.sent_tokenize(article)
                ff.write("\n".join(res))