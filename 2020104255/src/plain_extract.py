import os
import os.path as osp
from glob import glob
from tqdm import tqdm


if __name__ == "__main__":
    cnt = 1
    texs = glob("../extra/Arxiv6K.part1/*/*.tex",recursive=True)
    for tex_path in tqdm(texs):
        save_path = "../extra/tex2plain/" + str(cnt) + ".txt"
        cmd = f"/Users/yyj/opt/anaconda3/bin/pandoc -f latex -t plain {tex_path} -o {save_path}"
        os.system(cmd)
        cnt +=1