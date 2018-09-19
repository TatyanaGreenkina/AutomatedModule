import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
import re
import pymorphy2
from collections import Counter

morph = pymorphy2.MorphAnalyzer()
tokenizer = RegexpTokenizer(r'\w+')

data_file = pd.read_csv('./material.csv', sep=';', encoding='cp1251', error_bad_lines=False)

data_file = data_file[['FullName']]

for column in data_file.columns:
    data_file[column]=data_file[column].astype('str')

words = list()
for i in range(data_file.shape[0]):
    for column in data_file.columns:
        data_file[column][i]=re.sub(r'\d+','', data_file[column][i])
        for word in tokenizer.tokenize(data_file[column][i]):
            if len(word) != 1:
                word = morph.parse(word)[0].normal_form
                words.append(word.lower())

count=[]
count.extend(Counter(words).most_common())
word_to_index = {word: i for i,(word,_) in enumerate(count)}

final_embeddings = []

with open('./embeddings.txt', 'r') as f:
    embed = f.readlines()

for i in range(len(embed)):
    row_embed = [float(num) for num in embed[i].replace('\n', '').split('\t')[1:]]
    final_embeddings.append(row_embed)

one_word_row = list()

for i in range(len(data_file['FullName'])):
    a=tokenizer.tokenize(re.sub(r'\d+','',data_file['FullName'][i]))
    count_word =0
    for word in a:
        if(len(word))==1:
            count_word+=1
        if count_word == len(a):
            one_word_row.append(i)

rows = np.arange(len(data_file['FullName']))
rows = np.delete(rows, one_word_row)

final_embeddings_sent = np.zeros([len(data_file['FullName']), 64])

for i in rows:
    sum_vec=np.zeros(64)
    num_element = 0
    for word in tokenizer.tokenize(re.sub(r'\d+','', data_file['FullName'][i])):
        if len(word) !=1:
            num_element +=1
            word = morph.parse(word)[0].normal_form
            sum_vec = sum_vec +final_embeddings[word_to_index.get(word)]
    sent_vec = sum_vec/num_element
    final_embeddings_sent[i]=sent_vec

with open('final_embeddings.txt', 'w') as f:
    for n in rows:
        s = '\t'.join([str(num) for num in final_embeddings_sent[n]])
        f.write(s+'\n')
with open('mentadata_sent.txt', 'w') as f:
    for n in rows:
        f.write(data_file['FullName'][n] + '\n')
