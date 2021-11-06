# the answer 1 of the task 1:
import matplotlib.pyplot as plt
from elasticsearch import Elasticsearch
from image_match.elasticsearch_driver import SignatureES
from PIL import Image
import os.path as osp

if __name__ == "__main__":
    es = Elasticsearch()
    ses = SignatureES(es)
    # searched_image_path = "/Users/yyj/Downloads/Arxiv6K/Arxiv6K.part5/2001.07076/xin.jpg"
    # searched_image_path = "/Users/yyj/Downloads/Arxiv6K/Arxiv6K.part1/2011.07509/1_6.png"
    #
    searched_image_path = "/Users/yyj/Downloads/Arxiv6K/Arxiv6K.part1/2011.12956/Figures/robustness/test_param/2_Parametric_rob_nom_eta_magnitude.png"
    res = ses.search_image(searched_image_path)

    names = []
    for i,item in enumerate(res):
        if len(names) > 5:
            break
        basepath = osp.basename(item["path"])
        if basepath not in names:
            names.append(basepath)
        else:
            continue
        img = Image.open(item["path"]).convert('RGB')
        print(item)
        plt.title("score:"+str(item["score"]))
        plt.imshow(img)
        plt.show()
