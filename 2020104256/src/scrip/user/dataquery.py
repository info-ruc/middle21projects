# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy
import os, re
import pandas
import PyPDF2
import textract
from py2neo import Graph, Node, Relationship
from neo4j import GraphDatabase
import bibtexparser
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import *
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import convert_to_unicode
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
import user.dataset


def text_to_words(text):
    word_list_text = []

    text = text.replace('-\n', '')  # to one line
    text = text.replace('\n', '')
    words = text.split(' ')  # to words

    punctuations = {',', ';', '.', '!', '?', '\"', '\'', '(', ')'}
    pattern_hyphen = re.compile('[a-z]+-[a-z]+')  # pattern of words contain hyphens

    for word in words:
        if len(word) >= 2 or word in {'a', 'A', 'I'}:
            if word[0] in punctuations:
                word = word[1:]
            if word[-1] in punctuations:
                word = word[:-1]

            word = word.lower()  # to lower case
            if word.isalpha():
                word_list_text.append(word)
            elif pattern_hyphen.match(word):
                word_list_text.append(word)

    return word_list_text


def pdf_to_words(path, flag):
    pdf_file = open(path, mode='rb')  # 以二进制读模式打开
    print('current pdf path:  ', path, '\n')

    parser = PDFParser(pdf_file)  # 用文件对象来创建一个pdf文档分析器
    doc = PDFDocument(parser)  # 创建一个pdf文档
    parser.set_document(doc)

    word_list = []
    text_list = []

    if not doc.is_extractable:
        return word_list
        # raise PDFTextExtractionNotAllowed
    else:
        rsrcmgr = PDFResourceManager()  # 创建pdf资源管理器，来管理共享资源
        laparams = LAParams()  # 创建一个pdf设备对象
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)  # 创建一个pdf解释器对象

        # in each page
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)
            layout = device.get_result()  # 接受该页面的LTPage对象

            # in each layout
            for content in layout:
                if isinstance(content, LTTextBox):
                    text = content.get_text()
                    text_list.append(text)
    if flag == 1:
        # texts to words
        for text in text_list:
            word_list_text = text_to_words(text)
            word_list += word_list_text
        ser_words = pandas.Series(word_list)
        df_words = ser_words.reset_index()
        df_words.columns = ['index', 'word']
        return df_words
    else:
        return text_list


def get_key_words(df_words):
    vect = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    vect_transform = vect.fit_transform(df_words)

    # First approach of creating a dataframe of weight & feature names

    vect_score = numpy.asarray(vect_transform.mean(axis=0)).ravel().tolist()
    vect_array = pandas.DataFrame({'term': vect.get_feature_names(), 'weight': vect_score})
    vect_array.sort_values(by='weight', ascending=False, inplace=True)
    return vect_array


def title_key(title, dataframe):
    titles = []
    relations = []
    for i in range(len(dataframe)):
        titles.append(title)
        relations.append('key')

    datas = {'title': titles, 'relation': relations, 'key': dataframe}
    csv = pandas.DataFrame(data=datas)
    csv.to_csv("./data/title_key.csv", mode='a', header=False, index=False, sep=',')


def title_author(dataframe):
    csv = pandas.DataFrame(data=dataframe)
    csv.to_csv("./data/title_author.csv", mode='a', header=False, index=False, sep=',')


def file_name(file_dir):
    for dir in os.listdir(file_dir):
        print(dir)
        for root2, dirs2, files2 in os.walk(file_dir + "/" + dir):
            for file in files2:
                info = os.path.splitext(file)
                if os.path.splitext(file)[1] == '.pdf' and ('fig' not in os.path.splitext(file)[0]) and (
                        'Fig' not in os.path.splitext(file)[0]) \
                        and ('fig' not in root2) and ('Fig' not in root2) and (
                        'example' not in os.path.splitext(file)[0]) and ('graph' not in os.path.splitext(file)[0]) \
                        and ('img' not in root2) and ('Graph' not in os.path.splitext(file)[0]) and (
                        'square' not in os.path.splitext(file)[0]) and ('alns' not in os.path.splitext(file)[0]) \
                        and ('chart' not in os.path.splitext(file)[0]) and ('Chart' not in os.path.splitext(file)[0]):
                    print(file)
                    # pdf = open(root2 + "/" + file, mode='rb')
                    result = pdf_to_words(root2 + "/" + file, 2)
                    print(result)
                    if result and len(result) > 1 and '1\n' not in result and 'A\n' not in result:
                        tf_dataframe = get_key_words(result)
                        print(tf_dataframe.iloc[:10, :])
                        title_key(os.path.splitext(file)[0], tf_dataframe.iloc[:10, :]['term'])
                    # ------------------------------------
                    # PyPDF2
                    # ------------------------------------
                    # pdfdoc = PyPDF2.PdfFileReader(pdf)
                    # print(pdfdoc.numPages)
                    # # p=pdfdoc.getPage(2)
                    # # text = p.extractText()
                    # # print(text)
                    # num_pages = pdfdoc.numPages
                    # count = 0
                    # text = ""
                    # # while 循环会读取每一页
                    # while count < num_pages:
                    #     pageObj = pdfdoc.getPage(count)
                    #     count += 1
                    #     text += pageObj.extractText()
                    # print(text)
                    # ------------------------------------
                    # PyPDF2
                    # ------------------------------------
                    # parser = PDFParser(path)
                    # document = PDFDocument(parser)
                    # parser.set_document(document)
                    # document.set_parser(parser)
                    #
                    # document.initialize()
                    # rsrcmgr = PDFResourceManager()
                    # laparams = LAParams()
                    # device = PDFPageAggregator(rsrcmgr, laparams=laparams)
                    # interpreter = PDFPageInterpreter(rsrcmgr, device)
                    #
                    # for page in PDFPage.create_pages(document):
                    #     interpreter.process_page(page)
                    #     layout = device.get_result()
                    #     for x in layout:
                    #         if isinstance(x, LTTextBoxHorizontal):
                    #             with open('%s' % (root2 + "/pdf_trans_" + file), 'a') as f:
                    #                 results = x.get_text().encode('utf-8')
                    #                 f.write(results + "\n")
                elif os.path.splitext(file)[1] == '.bib' and os.path.splitext(file)[0][:3] != 'gbk':
                    print(root2 + "/" + file)
                    path = root2 + "/gbk_" + file
                    lines = [l.encode('gbk', errors='ignore').decode('gbk') for l in
                             open(root2 + "/" + file, encoding='gb18030', errors='ignore') if
                             l.strip().find("month", 0, 5) != 0 and l.strip().find("Month", 0, 5) != 0
                             and l.strip().find("booktitle", 0, 9) != 0 and l.strip().find("OPTacknowledgement", 0,
                                                                                           18) != 0 and l.strip().find(
                                 "nomonth", 0, 18) != 0 and l.strip().find("NOMONTH", 0, 18) != 0
                             and l.strip().find("MONTH", 0, 5) != 0 and l.strip().find("publisher", 0,
                                                                                       15) != 0 and l.strip().find(
                                 "ieee_j_edl") != 0]
                    fd = open(path, "w")
                    fd.writelines(lines)
                    fd.close()
                    # file = open(root2 + "/" + file, encoding='gb18030', errors='ignore')
                    # file2 = open(path, 'w')
                    # file2.write(file.read())
                    # file.close()
                    # file2.close()
                    with open(path) as bibfile:
                        parser = BibTexParser()  # 声明解析器类
                        parser.customization = convert_to_unicode  # 将BibTeX编码强制转换为UTF编码
                        bibdata = bibtexparser.load(bibfile)  # 通过bp.load()加载
                        for l in bibdata.entries:
                            if 'author' in l and 'title' in l:
                                print(l['author'])
                                print(l['title'])
                                d = {'author': [l['author']], 'relation': ['author'], 'title': [l['title']]}
                                title_author(d)


# class PaperExtractor(object):
#     def __init__(self):
#         super(PaperExtractor, self).__init__()
#         self.graph = Graph(
#             host="127.0.0.1",
#             http_port="7474",
#             user="neo4j",
#             password="123123"
#         )
#
#         self.author = []
#         self.title = []
#         self.key = []
#
#         self.write = []
#         self.keyword = []
#
#         def triples(self,path):
#             print('读取并转换csv')
#             datas = pandas.read_csv(path)
#             tri = {}
#             for data in datas:
#
#                 if 'title' in datas.columns:
#                     tri['title'] = data.author
#
#                 if 'rela' in datas.columns:
#                     tri['rela'] = data.rela
#                 if 'title' in datas.columns:
#                     tri['title'] = data.title
#                 if 'key' in datas.columns:
#                     tri['key'] = data.key

driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "123123"))


def load_data():
    with driver.session() as session:
        session.run("""MATCH ()-[r]->() DELETE r""")
        session.run("""MATCH (r) DELETE r""")

        session.run("""LOAD CSV WITH HEADERS FROM "file:///title_author.csv" AS csv 
                        MERGE (u:Author {name: csv.author})
                        MERGE (m:Paper {title: csv.title})
                                CREATE (u)-[:Rela {rela: csv.rela}]->(m)""")

        session.run("""LOAD CSV WITH HEADERS FROM "file:///title_key.csv" AS csv 
                        MERGE (m:Key {keyword: csv.keyword})
                        MERGE (u:Paper {title: csv.title})
                                CREATE (u)-[:Rela {rela: csv.rela}]->(m)""")


def queries(chose,query):
    # while True:
    #     chose = int(input("查询模式（1.查作者，2.查文献，3.查领域）："))
        if chose == 1:
            author = input("查询的作者名字：")
            with driver.session() as session:
                q = session.run(f"""MATCH (a:Author {{name:{query}}} )-[Rela:rela]->(p:Paper)
                RETURN p.title as title""")
                result = []
                keyword = []
                for i in enumerate(q):
                    result.append(i["title"])
                    q2 = session.run(f"""MATCH (p:Paper {{title:{i["title"]}}} )-[Rela:rela]->(k:Key)
                            RETURN k.keyword as keyword""")
                    for j in enumerate(q2):
                        keyword.append(j["keyword"])

                print("该作者的作品：")
                print(result)
                print("该作者的领域：")
                print(keyword)
                r = {"该作者的作品：" + result + "\n该作者的领域：" + keyword}
                return r
        elif chose == 2:
            title = input("查询的文献名字：")
            with driver.session() as session:
                q = session.run(f"""MATCH (a:Author )-[Rela:rela]->(p:Paper {{title:{query}}})
                RETURN a.name as name""")
                result = []
                keyword = []
                for i in enumerate(q):
                    result.append(i["name"])
                    q2 = session.run(f"""MATCH (p:Paper {{title:{query}}} )-[Rela:rela]->(k:Key)
                            RETURN k.keyword as keyword""")
                    for j in enumerate(q2):
                        keyword.append(j["keyword"])

                print("该作品的作者：")
                print(result)
                print("该作品的领域：")
                print(keyword)
                r = {"该作品的作者：" + result + "\n该作品的领域：" + keyword}
                return r
        elif chose == 3:
            keyword = input("查询的领域：")
            with driver.session() as session:
                q = session.run(f"""MATCH (p:Paper  )-[Rela:rela]->(k:Key  {{keyword:{query}}})
                        RETURN p.title as title""")
                result = []
                for i in enumerate(q):
                    result.append(i["title"])
                print("该领域的作品：")
                print(result)
                r = {"该领域的作品：" + result }
                return r
        else:
            print('输入异常')
            r = {'输入异常'}
            return r


# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# data = pandas.read_csv('./data/title_author.csv',encoding='gbk')
# file_name("./data")
# load_data()

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------
# file_name("D:\baiduyun\Arxiv6K")
# # #
# graph = Graph('http://localhost:7474', name='neo4j', password='123123')
#
# ##创建结点
# test_node_1 = Node(label='Paper', name='asd')
# test_node_2 = Node(label='Author', name='123')
# test_node_3 = Node(label='Keyword', name='cccccc')
# graph.create(test_node_1)
# graph.create(test_node_2)
# graph.create(test_node_3)
#
# ##创建关系
# # 分别建立了test_node_1指向test_node_2和test_node_2指向test_node_1两条关系，关系的类型为"丈夫、妻子"，两条关系都有属性count，且值为1。
# node_1_zhangfu_node_1 = Relationship(test_node_1, 'write', test_node_2)
# node_1_zhangfu_node_1['count'] = 1
# node_2_qizi_node_1 = Relationship(test_node_1, 'key', test_node_3)
#
# node_2_qizi_node_1['count'] = 1
#
# graph.create(node_1_zhangfu_node_1)
# graph.create(node_2_qizi_node_1)
#
# print(graph)
# print(test_node_1)
# print(test_node_2)
# print(node_1_zhangfu_node_1)
# print(node_2_qizi_node_1)
