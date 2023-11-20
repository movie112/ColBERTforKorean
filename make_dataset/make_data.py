import pandas as pd
import numpy as np
import os
import json
from tqdm.auto import tqdm
from rank_bm25 import BM25Okapi

from kobert import KoBERTTokenizer

def load_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        dic = json.load(f)
    return dic['data']

large_dir1 = '../dataset/웹데이터기반_한국어말뭉치_데이터/01.데이터/1.Training/원천데이터/'
large_dir2 = '../dataset/웹데이터기반_한국어말뭉치_데이터/01.데이터/2.Validation/원천데이터/'
file_lst = []
dir_lst = []
for i in range(1, 10):
    dir_lst.append(large_dir1+'t0'+str(i))
    dir_lst.append(large_dir2+'t0'+str(i))
for i in range(10,18):
    dir_lst.append(large_dir1+'t'+str(i))
    dir_lst.append(large_dir2+'t'+str(i))
for dir in dir_lst:
    for (root, directories, files) in os.walk(dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_lst.append(file_path)
large_data = []
for file_path in file_lst:
    large_data.append(load_file(file_path))


df_train20 = pd.read_csv('../dataset/df_train20.csv')
df_test20 = pd.read_csv('../dataset/df_test20.csv')
df_200 = pd.read_csv('../dataset/df_200.csv')
df_50 = pd.read_csv('../dataset/df_50.csv')
df_20 = pd.read_csv('../dataset/df_20.csv')

df_train20.columns = ['qid', 'pid', 'q', 'p']
df_test20.columns = ['qid', 'pid', 'q', 'p']
df_200.columns = ['qid', 'pid', 'q', 'p']
df_50.columns = ['qid', 'pid', 'q', 'p']
df_20.columns = ['qid', 'pid', 'q', 'p']

# collection(passages)
collec = pd.DataFrame({'pid': df_200['pid'], 'p':df_200['p']}).sort_values(by=['pid'], axis=0)
collec = collec[~collec.duplicated()]

p_large = []
i = 0
lc = []
for data in tqdm(large_data):
    for text in data['SJML']['text']:
        content = text['content'].replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
        lc.append(len(content))
        if len(content) > 1000 or len(content) < 40:
            continue
        p_large.append(content)
collec_large = pd.DataFrame({'p':p_large})
collec_large = collec_large[~collec_large.duplicated(subset='p')].reset_index(drop=True)
collec_large = collec_large.sample(frac=1, random_state=46).reset_index(drop=True)

collec_l=collec_large[:int(len(collec))]
del collec_large
pid_large = list(range(collec['pid'].max()+1, len(collec_l)+collec['pid'].max()+1))
collec_l.insert(0, 'pid', pid_large)
collec_l.sort_values(by=['pid'], axis=0)

collec_all = pd.concat([collec, collec_l])
collec_all = collec_all.reset_index(drop=True)
collec_all = collec_all[~collec_all.duplicated(subset='p')].reset_index(drop=True)
collec_all.to_csv('../dataset/collection.csv', index=False)


# queries
queries20 = pd.DataFrame({'qid': df_test20['qid'], 'q' :df_test20['q']})
queries20.to_csv('../dataset/queries20.csv', index=False)

# qrels
lst0 = [0 for i in range(len(df_test20))]
lst1 = [1 for i in range(len(df_test20))]

qrels20 = pd.DataFrame({'qid':df_test20['qid'], 'zero':lst0, 'pid':df_test20['pid'], 'one':lst1})
qrels20.to_csv('../dataset/qrels20.csv', index=False)

# top
tok = KoBERTTokenizer.from_pretrained('monologg/kobert')
passages = collec_all['p'].tolist()

toked_p = [tok.tokenize(doc) for doc in tqdm(passages)]
bm25 = BM25Okapi(toked_p)
del toked_p

# top_20
top20 = pd.DataFrame(columns={'qid', 'pid', 'q', 'p'})
NUM = 1000

for i in tqdm(range(len(df_test20))):
    qid_lst = [df_test20.iloc[i]['qid'] for j in range(NUM)]
    query = df_test20.iloc[i]['q']
    q_lst = [query for j in range(NUM)]
       
    toked_q = tok.tokenize(query)
    doc_scores = bm25.get_scores(toked_q)
    
    scores = pd.DataFrame({'pid': collec['pid'],'p':collec['p'], 'score':doc_scores})
    scores=scores.sort_values(by='score', axis=0, ascending=False)
    scores = scores.iloc[:NUM, [0, 1]]
    scores.insert(0, 'qid', qid_lst)
    scores.insert(2, 'q', q_lst)
    
    top20 = pd.concat([top20, scores])
top20 = top20.reset_index(drop=True)
top20.to_csv('../dataset/top20.csv', index=False)

# triples_20
triples20 = pd.DataFrame(columns={'qid', 'pid_pos', 'pid_neg', 'q', 'p_pos', 'p_neg', 'score'})
q_lst = [df_train20.iloc[i]['q'] for i in range(len(df_train20))]
p_pos_lst = [df_train20.iloc[i]['p'] for i in range(len(df_train20))]
qid_lst = [df_train20.iloc[i]['qid'] for i in range(len(df_train20))]
pid_pos_lst = [df_train20.iloc[i]['pid'] for i in range(len(df_train20))]
NUM = 20

for i in tqdm(range(len(q_lst))):

    toked_q = tok.tokenize(q_lst[i])
    doc_scores = bm25.get_scores(toked_q)

    scores = pd.DataFrame({'pid_neg': collec['pid'],'p_neg':collec['p'], 'score':doc_scores})
    scores=scores.sort_values(by='score', axis=0, ascending=False)
    scores = scores.iloc[:NUM, [0, 1]]
    scores.insert(0, 'pid_pos',[pid_pos_lst[i] for j in range(NUM)])
    scores.insert(0, 'qid',[qid_lst[i] for j in range(NUM)])
    scores.insert(3, 'p',[p_pos_lst[i] for j in range(NUM)])
    scores.insert(3, 'q', [q_lst[i] for j in range(NUM)])

    triples20 = pd.concat([triples20, scores])
triples20 = triples20.reset_index(drop=True)
triples20.to_csv('../dataset/triples20.csv', index=False)
del triples20

# triples_50
triples50 = pd.DataFrame(columns={'qid', 'pid_pos', 'pid_neg', 'q', 'p_pos', 'p_neg', 'score'})
q_lst = [df_50.iloc[i]['q'] for i in range(len(df_50))]
p_pos_lst = [df_50.iloc[i]['p'] for i in range(len(df_50))]
qid_lst = [df_50.iloc[i]['qid'] for i in range(len(df_50))]
pid_pos_lst = [df_50.iloc[i]['pid'] for i in range(len(df_50))]
NUM = 20

for i in tqdm(range(len(q_lst))):
    
    toked_q = tok.tokenize(q_lst[i])
    doc_scores = bm25.get_scores(toked_q)

    scores = pd.DataFrame({'pid_neg': collec['pid'],'p_neg':collec['p'], 'score':doc_scores})
    scores=scores.sort_values(by='score', axis=0, ascending=False)
    scores = scores.iloc[:NUM, [0, 1]]
    scores.insert(0, 'pid_pos',[pid_pos_lst[i] for j in range(NUM)])
    scores.insert(0, 'qid',[qid_lst[i] for j in range(NUM)])
    scores.insert(3, 'p',[p_pos_lst[i] for j in range(NUM)])
    scores.insert(3, 'q', [q_lst[i] for j in range(NUM)])

    triples50 = pd.concat([triples50, scores])
triples50 = triples50.reset_index(drop=True)
triples50.to_csv('../dataset/triples50.csv', index=False)
del triples50

# triples_200
triples200 = pd.DataFrame(columns={'qid', 'pid_pos', 'pid_neg', 'q', 'p_pos', 'p_neg', 'score'})
q_lst = [df_200.iloc[i]['q'] for i in range(len(df_200))]
p_pos_lst = [df_200.iloc[i]['p'] for i in range(len(df_200))]
qid_lst = [df_200.iloc[i]['qid'] for i in range(len(df_200))]
pid_pos_lst = [df_200.iloc[i]['pid'] for i in range(len(df_200))]
NUM = 20

for i in tqdm(range(len(q_lst))):

    toked_q = tok.tokenize(q_lst[i])
    doc_scores = bm25.get_scores(toked_q)

    scores = pd.DataFrame({'pid_neg': collec['pid'],'p_neg':collec['p'], 'score':doc_scores})
    scores=scores.sort_values(by='score', axis=0, ascending=False)
    scores = scores.iloc[:NUM, [0, 1]]
    scores.insert(0, 'pid_pos',[pid_pos_lst[i] for j in range(NUM)])
    scores.insert(0, 'qid',[qid_lst[i] for j in range(NUM)])
    scores.insert(3, 'p',[p_pos_lst[i] for j in range(NUM)])
    scores.insert(3, 'q', [q_lst[i] for j in range(NUM)])

    triples200 = pd.concat([triples200, scores])
triples200 = triples200.reset_index(drop=True)
triples200.to_csv('../dataset/triples200.csv', index=False)
del triples200



