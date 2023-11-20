import json
import numpy as np
import pandas as pd
import os
import torch
import random
import re
from tqdm.auto import tqdm

from rank_bm25 import BM25Okapi

from collections import Counter
# import matplotlib.pyplot as plt

from tokenizer import KoBertTokenizer

def load_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        dic = json.load(f)
    return dic['data']

def processing(df, start_pid=0):
    # q-p 중복되는 행 삭제collec = collec[~collec.duplicated()]
    df = df[~df.duplicated(subset={'q', 'p'})].reset_index(drop=True)

    # query 중복되는 행
    df_dup = df[df.duplicated(subset='q', keep=False)].sort_values(by=['q'], axis=0)

    # qeury 중복이면 qid 같게 만들어줌 
    pre_row = df_dup.iloc[0] # 첫 행
    for i in range(len(df_dup)):
        if pre_row['q'] == df_dup['q'].iloc[i]:
            df.loc[df['qid'] == df_dup.iloc[i]['qid'], 'qid'] = pre_row['qid']
        else:
            pre_row = df_dup.iloc[i]
    # add pid
    df = df.sort_values(by=['p'], axis=0)
    pre_i = 0
    pid = [start_pid]
    for i in tqdm(range(1, len(df))):
        if df.iloc[i-1]['p'] != df.iloc[i]['p']:
            pre_i += 1
        pid.append(pre_i + start_pid)
    df.insert(1, 'pid', pid)
    df = df.sample(frac=1, random_state=46).reset_index(drop=True)
    
    return df

def cut_df(df, max_len=50):
    df_cut = df.copy()
    drop_ind = []
    for i in range(len(df)):
        if len(df.loc[i]['q']) > max_len:
            drop_ind.append(i)
    df_cut = df_cut.drop(drop_ind).reset_index(drop=True)
    return df_cut

# load_Datasets
cs_path = '../dataset/일반상식/02_squad_질문_답변_제시문_말뭉치/ko_wiki_v1_squad.json'
cs_data = load_file(cs_path)

mrc_path1 = '../dataset/기계독해분야/normal/ko_nia_normal_squad_all.json'
mrc_path2 = '../dataset/기계독해분야/noanswer/ko_nia_noanswer_squad_all.json'
mrc_path3 = '../dataset/기계독해분야/clue/ko_nia_clue0529_squad_all'
mrc_data1 = load_file(mrc_path1)
mrc_data2 = load_file(mrc_path2)
mrc_data3 = load_file(mrc_path3)
mrc_data = mrc_data1 + mrc_data2 + mrc_data3
del mrc_data1, mrc_data2, mrc_data3

book_path1 = '../dataset/도서자료_기계독해/Training/book_train_220419_add.json'
book_path2 = '../dataset/도서자료_기계독해/Validation/book_dev.json'
book_data1 = load_file(book_path1)
book_data2 = load_file(book_path2)
book_data = book_data1 + book_data2
del book_data1, book_data2


######################
cs_id = []
cs_p = []
cs_q = []
for i in tqdm(range(len(cs_data))):
    context = cs_data[i]['paragraphs'][0]['context'].replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
    question = cs_data[i]['paragraphs'][0]['qas'][0]['question'].replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
    if len(context) > 1000: continue
    if len(question) < 4: continue
        
    cs_q.append(question)
    cs_p.append(context)
    cs_id.append(100000000+i)
cs_df = pd.DataFrame({'qid':cs_id, 'q':cs_q, 'p':cs_p})
cs_df = processing(cs_df, start_pid=0)

mrc_id = []
mrc_q = []
mrc_p = []
counter = 0
for i in tqdm(range(len(mrc_data))):
    context = mrc_data[i]['paragraphs'][0]['context'].replace('\n', ' ').replace('\t', ' ')     
    qas = mrc_data[i]['paragraphs'][0]['qas']
    if len(context) > 1000: continue
    
    for j in range(len(qas)):
        question = qas[j]['question'].replace('\n', ' ').replace('\t', ' ')
        if len(question) < 4: continue
        mrc_q.append(question)
        mrc_p.append(context)
        mrc_id.append(cs_df['qid'].max()+1+counter) # id 다 unique하도록
        counter += 1     
mrc_df = pd.DataFrame({'qid':mrc_id, 'q':mrc_q, 'p':mrc_p})
mrc_df = processing(mrc_df, start_pid=cs_df['pid'].max()+1)

book_id = []
book_q = []
book_p = []
counter = 0
for i in tqdm(range(len(book_data))):
    for k in range(len(book_data[i]['paragraphs'])):
        context = book_data[i]['paragraphs'][k]['context'].replace('\n', ' ').replace('\t', ' ')
        qas = book_data[i]['paragraphs'][k]['qas']
        if len(context) > 1000: continue
            
        for j in range(len(qas)):
            question = qas[j]['question'].replace('\n', ' ').replace('\t', ' ')
            if len(question) < 4: continue
            book_q.append(question)
            book_p.append(context)
            book_id.append(mrc_df['qid'].max()+1+counter)
            counter += 1
book_df = pd.DataFrame({'qid':book_id, 'q':book_q, 'p':book_p})
book_df = processing(book_df, start_pid=mrc_df['pid'].max()+1)

# df_200
N = len(cs_df)
df_all = pd.concat([cs_df, mrc_df, book_df], axis=0).reset_index(drop=True)
df_200 = pd.concat([cs_df[:N], mrc_df[:N], book_df[:N]]).reset_index(drop=True)

# df_50
cs_df50 = cut_df(cs_df, max_len=50)
mrc_df50 = cut_df(mrc_df, max_len=50)
book_df50 = cut_df(book_df, max_len=50)
N = len(cs_df50)
df_50 = pd.concat([cs_df50[:N], mrc_df50[:N], book_df50[:N]], axis=0).reset_index(drop=True)

# df_20
cs_df20 = cut_df(cs_df50, max_len=20)
mrc_df20 = cut_df(mrc_df50, max_len=20)
book_df20 = cut_df(book_df50, max_len=20)
N = len(cs_df20)
df_20 = pd.concat([cs_df20[:N], mrc_df20[:N], book_df20[:N]], axis=0).reset_index(drop=True)

# train/test split
df_20 = df_20.sample(frac=1, random_state=46).reset_index(drop=True)
df_test20 = df_20.iloc[:3000, :]
df_train20 = df_20.iloc[3000:, :]

df_200 = df_200.sample(frac=1, random_state=46).reset_index(drop=True)
df_50 = df_50.sample(frac=1, random_state=46).reset_index(drop=True)

# test와 겹치면 행 삭제
df_200 = df_200[~df_200['q'].isin(df_test20['q'])].reset_index(drop=True)
df_50 = df_50[~df_50['q'].isin(df_test20['q'])].reset_index(drop=True)

df_200.to_csv('../dataset/df_200.csv', index=False)
df_50.to_csv('../dataset/df_50.csv', index=False)
df_20.to_csv('../dataset/df_20.csv', index=False)
df_train20.to_csv('../dataset/df_train20.csv', index=False)
df_test20.to_csv('../dataset/df_test20.csv', index=False)




