from collections import Counter
from nltk.collocations import *
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pymorphy2
import re


class Preprocessing(object):
    def __init__(self,
                 data_path,
                 special_token):

        self.morph = pymorphy2.MorphAnalyzer()
        # self.tokenizer = RegexpTokenizer(r'\w+/\w+|\w+-\w+|\w+')
        self.tokenizer = RegexpTokenizer(r'\w+')

        with open(data_path, 'r') as f:
            lines = [line.replace('\n', '').split('\t') for line in f.readlines()]
            self.rows = {elem[0]: elem[1:] for elem in lines}

        elem = [len(i) for i in self.rows.keys()]
        self.length = max(elem)

        self.word_index = {}
        self.words = {}
        self.words_data = []

        for key in self.rows.keys():
            dict_pos = {word: list() for word in set(self.rows[key])}
            for (pos, word) in enumerate(self.rows[key]):
                dict_pos[word].append(pos)
            self.rows[key] = dict_pos

        for elem_key in self.rows.keys():
            for word_key in self.rows[elem_key].keys():
                if word_key in self.word_index.keys():
                    self.word_index[word_key].update({elem_key: self.rows[elem_key][word_key]})
                else:
                    self.word_index[word_key] = {elem_key: self.rows[elem_key][word_key]}

        with open(r'./matrix/mentadata.txt', 'r') as f:
            self.data = [line.replace('\n', '') for line in f.readlines()]

        count = []
        count.extend(Counter(self.data).most_common())
        self.word_to_index = {special_token: 0}
        self.word_to_index.update({word: i+1 for i, (word, _) in enumerate(count)})
        self.index_to_word = dict(zip(self.word_to_index.values(), self.word_to_index.keys()))

    def searcher(self, elem):
        words = [i.lower() for i in self.tokenizer.tokenize(re.sub(r'\d+', '', elem)) if len(i) > 1]
        words = [self.morph.parse(i)[0].normal_form for i in words if i == 'то' or i not in stopwords.words('russian')]
        words_split = []
        for i in words:
            try:
                elem_for_words = list(self.word_index[i].keys())
                if len(elem_for_words) > 1:
                    elem_for_words = [j for j in elem_for_words]
            except KeyError:
                elem_for_words = ['<UNK>']
            words_split.extend(elem_for_words)
        elem = [i for (i, _) in Counter(words_split).most_common()]
        return elem[0]

    def prepare_data(self, elem):
        # TODO split elem with tokenizer and morphanalyzer
        elem_in_dict = self.searcher(elem)
        if elem_in_dict != '<UNK>':
            elem_in_dict = [i.lower() for i in self.tokenizer.tokenize(re.sub(r'\d+', '', elem_in_dict)) if len(i) > 1]
            elem_in_dict = [self.morph.parse(i)[0].normal_form for i in elem_in_dict if i == 'то'
                            or i not in stopwords.words('russian')]
        return elem_in_dict

    def create_train_data(self, batch):
        # TODO create a data to teaching CNN
        words_in_dict = []
        for elem in batch:
            words_in_dict.append([self.word_to_index[i] for i in self.prepare_data(elem) if i not in '<UNK>'])
        return words_in_dict

    def create_emb_dict(self):
        embeddings = []
        with open('./matrix/embeddings.txt', 'r', encoding='utf8') as f:
            emb_lines = f.readlines()
            for i in range(len(emb_lines)):
                embeddings.append([float(num) for num in emb_lines[i].replace('\n', '').split('\t')[1:]])
        embeddings = np.array(embeddings)
        final_embeddings = np.concatenate((np.zeros((1, embeddings.shape[1])), embeddings), axis=0)
        elems = {i: [self.word_to_index[j] for j in self.rows[i].keys() if not re.findall(r'\d', j) and j] for i in self.rows.keys()}
        embed = {key: np.mean(final_embeddings[elems[key]], axis=0) for key in elems.keys()}
        return final_embeddings, embed

    def search_simil_elems(self, elem, num):
        index_words = self.create_train_data([elem])
        word_emb, embed_dict = self.create_emb_dict()
        search_result = []
        vec1 = np.mean(word_emb[index_words[0]], axis=0)
        for key in embed_dict.keys():
            vec2 = embed_dict[key]
            simil = 1 - spatial.distance.cosine(vec1, vec2)
            if simil > 0.9:
                search_result.append([key, simil])
        search_result.sort(key=lambda x:x[1], reverse=True)
        return search_result[:num]


#=============================== Experiment ==========================================

# prep = Preprocessing(r'./dict', special_token='<UNK>')
# print(prep.create_train_data(['Бланк для записи потенциалов',
# 'Профиль ПС 75х50 3 м Кнауф 0,60 мм', 'Шайба 0730 107 966 01']))
#
#
# print(prep.prepare_data('лиаз топливо болт'))
# data_file = pd.read_csv('./Data/material.csv', sep=';', encoding = 'cp1251', error_bad_lines=False,
#                         low_memory=False)[['FullName']]
#
# data_file = data_file[['FullName']].astype('str')
#
# morph = pymorphy2.MorphAnalyzer()
# tokenizer = RegexpTokenizer(r'\w+/\w+|\w+-\w+|\w+')
#
# rows = [row[0] for row in data_file.values[:10]]

# print(rows[:10])

# length = {word: len(word) for word in rows}
# length = sorted(length.items(), key=lambda item: item[1], reverse=True)

# words = []
# for row in tqdm(rows):
#     words.extend([i.lower() for i in tokenizer.tokenize(re.sub(r'\d+', '', row)) if len(i) > 1])
# words = [morph.parse(i)[0].normal_form for i in tqdm(words) if i == 'то' or i not in stopwords.words('russian')]

# nomen = pd.read_csv('./Data/nomenklatura.csv', sep=';', encoding = 'cp1251', error_bad_lines=False,
#                         low_memory=False)[['FullName']]
#



# with open('dict', 'a') as f:
#     for key in words.keys():
#         f.write('{}\t{}\n'.format(key, '\t'.join(words[key])))

# for key in tqdm(words.keys()):
#     dict_pos = {word: list() for word in set(words[key])}
#     for (pos, word) in enumerate(words[key]):
#         dict_pos[word].append(pos)
#     words[key] = dict_pos
#
#
# word_index = {}
# for elem_key in tqdm(words.keys()):
#     for word_key in words[elem_key].keys():
#         if word_key in word_index.keys():
#             word_index[word_key].update({elem_key: words[elem_key][word_key]})
#         else:
#             word_index[word_key] = {elem_key: words[elem_key][word_key]}
# print(word_index.keys())



# def searcher(elem):
#     elem_word = [morph.parse(i)[0].normal_form.lower() for i in tokenizer.tokenize(re.sub(r'\d+', '', elem)) if len(i) > 1
#             and (i == 'то' or i not in stopwords.words('russian'))]
#     for word in elem_word:
#         try:
#             word_index[word]
#         except KeyError:
#             with open(r'./log_nomen.txt', 'a') as f:
#                 f.write('KeyError ' + word + '--' + elem + '\n')


#===================================Statistical indicators=================================
# word_freq = Counter(words).most_common()
#
# bigrams = Counter(nltk.bigrams(words)).most_common()
#==========================================================================================


# bigram_measures = nltk.collocations.BigramAssocMeasures()
# f_2 = nltk.collocations.BigramCollocationFinder.from_words(words)
# bigrams_pmi = f_2.score_ngrams(bigram_measures.pmi)


# trigram_measures = nltk.collocations.TrigramAssocMeasures()
# f_3 = nltk.collocations.TrigramCollocationFinder.from_words(words)
# trigram_pmi = f_3.score_ngrams(trigram_measures.pmi)
#
# df1 = pd.DataFrame({
#     'elem': [i for (i, _) in length],
#     'len': [i for (_, i) in length]
# })
#
# df2 = pd.DataFrame({
#     'word': [i for (i, _) in tqdm(word_freq)],
#     'freq': [i for (_, i) in tqdm(word_freq)]
# })
#
# df3 = pd.DataFrame({
#     'bigram': [i for (i, _) in tqdm(bigrams)],
#     'freq': [i for (_, i) in tqdm(bigrams)]
# })
#
# df4 = pd.DataFrame({
#     'bigram_pmi': [i for (i, _) in tqdm(bigrams_pmi)],
#     'freq': [i for (_, i) in tqdm(bigrams_pmi)]
# })
#
# df5 = pd.DataFrame({
#     'trigram_pmi': [i for (i, k) in tqdm(trigram_pmi) if k == trigram_pmi[0][1]],
#     'freq': [i for (_, i) in tqdm(trigram_pmi) if i > i == trigram_pmi[0][1]]
# })
#
# writer = pd.ExcelWriter(r'./output_nomen.xlsx')
# df1.to_excel(writer, 'Length')
# df2.to_excel(writer, 'Freq of words')
# df3.to_excel(writer, 'Freq of bigrams')
# df4.to_excel(writer, 'Bigram collocation')
# df5.to_excel(writer, 'Trigram collocation')
# writer.save()
