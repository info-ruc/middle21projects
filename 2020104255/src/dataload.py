from elasticsearch import Elasticsearch
from image_match.elasticsearch_driver import SignatureES
import os
import os.path as osp
from glob import glob
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from time import sleep


def find_image_file(source_path, file_lst):
    """
    递归寻找 文件夹以及子目录的 图片文件。
    :param source_path: 源文件夹路径
    :param file_lst: 输出 文件路径列表
    :return:
    """
    image_ext = ['.jpg', '.JPG', '.PNG', '.png', '.jpeg', '.JPEG', '.bmp']
    for dir_or_file in os.listdir(source_path):
        file_path = os.path.join(source_path, dir_or_file)
        if os.path.isfile(file_path):  # 判断是否为文件
            file_name_ext = os.path.splitext(os.path.basename(file_path))  # 文件名与后缀
            if len(file_name_ext) < 2:
                continue
            if file_name_ext[1] in image_ext:  # 后缀在后缀列表中
                file_lst.append(file_path)
            else:
                continue
        elif os.path.isdir(file_path):  # 如果是个dir，则再次调用此函数，传入当前目录，递归处理。
            find_image_file(file_path, file_lst)
        else:
            pass
            # print('文件夹没有图片' + os.path.basename(file_path))
thread_count = 100

def add_image_of_arxiv_item(path):
    image_paths = []
    find_image_file(path, image_paths)
    for image_path in image_paths:
        try:
            ses.add_image(image_path)
        except Exception as e:
            print(e)

MAX_WORKSER = 20
REMAIN = 20

def finish(*args,**params):
    global REMAIN

    REMAIN = REMAIN + 1

if __name__ == "__main__":
    es = Elasticsearch()
    ses = SignatureES(es)
    extra_path = "/Users/yyj/PycharmProjects/middle21projects/2020104255/extra/"
    pool = ThreadPoolExecutor(max_workers=MAX_WORKSER)

    tasks = []

    for dir in os.listdir(extra_path):
        for i,arxiv_id in enumerate(tqdm(os.listdir(osp.join(extra_path,dir)))):
            path = osp.join(extra_path, dir, arxiv_id)
            if i==0:
                continue
            # add_image_of_arxiv_item(path)
            task = pool.submit(add_image_of_arxiv_item,path)
            REMAIN -= 1
            while REMAIN <= 0:
                sleep(5)

            task.add_done_callback(finish)
            tasks.append(task)


