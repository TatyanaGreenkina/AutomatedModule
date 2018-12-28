import numpy as  np
import pandas as pd
import embeddings
from tqdm import tqdm
import re

morph = pymorphy2.MorphAnalyzer()
tokenizer = RegexpTokenizer(r'\w+')

embeding_matrix = embeddings.Embedding()

data = pd.read_csv(r'./Data/labeled_data.csv', sep=';', encoding='cp1251', error_bad_lines=False)[['ЭталоннаяПозиция',
                                                                                                   'Номенклатура']]
for column in tqdm(data.columns):
    data[column] = [i.lower() for i in data[column]]
    embeding_matrix.fine_tuning(data[column])

error = len(data)*64
error_initial = 0
error_words_1 = error_initial
error_words_2 = 0

for i in tqdm(range(len(data))):
    try:
        word = data['ЭталоннаяПозиция'][i]
        label = data['Номенклатура'][i]
        error_initial = error_initial + (1 - embeding_matrix.rang(word, label))**2
    except KeyError:
        pass

while error >= 0.01:
    for i in tqdm(range(len(data))):
        try:
            word = data['ЭталоннаяПозиция'][i]
            label = data['Номенклатура'][i]
            rang = embeding_matrix.rang(word, label)
            update = list()
            if 1 - rang >= 0:
                sign = 1
            else:
                sign = -1
            em = embeding_matrix.emb_dict_sent[word]
            update.append([sign*embeding_matrix.cos(word, label)*x for x in em])
            for key in tqdm(embeding_matrix.emb_dict_sent.keys()):
                if key != word or label:
                    update.append([-1*embeding_matrix.cos(word, key)*x for x in em])
            update = np.mean(update, axis=0)
            update = [0.1*x for x in update/np.sum(np.power(update, 2))]
            emb = np.add(em, update)
            embeding_matrix.emb_dict_sent[word] = emb/np.sum(np.power(emb, 2))
            error_iter = 1 - embeding_matrix.rang(word, label)**2
        except KeyError:
            with open('log.txt', 'a') as f:
                f.write(word + '--' + label)
        error_words_2 = error_words_2 + error_iter
    error = (error_words_1 - error_words_2) / error_initial
    error_words_1 = error_words_2

with open('./matrix/tuning_embeddings.txt', 'a') as f:
    for n in embeding_matrix.emb_dict_sent.values():
        s = '\t'.join([str(num) for num in n])
        f.write(s+'\n')
