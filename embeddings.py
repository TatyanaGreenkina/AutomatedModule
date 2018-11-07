from scipy import spatial
import numpy as np
import re
import pymorphy2
from nltk.tokenize import RegexpTokenizer

morph = pymorphy2.MorphAnalyzer()
tokenizer = RegexpTokenizer(r'\w+')

class Embedding(object):
    def __init__(self):
        self.emb_dict_sent = {}
        self.emb_dict = {}
        self.answers = {}
        with open('./matrix/mentadata_sent.txt', 'r', encoding='utf8') as f:
            name_emb_sent = [row.replace('\n', '') for row in f.readlines()]

        with open('./matrix/final_embeddings.txt', 'r', encoding='utf8') as f:
            embed_sent = f.readlines()

        for i in range(len(embed_sent)):
            row_embed_sent = [float(num)
                         for num in embed_sent[i].replace('\n', '').split('\t')]
            self.emb_dict_sent[name_emb_sent[i]] = row_embed_sent

        with open('./matrix/mentadata.txt', 'r', encoding='utf8') as f:
            name_emb = [row.replace('\n', '') for row in f.readlines()]

        with open('./matrix/embeddings.txt', 'r', encoding='utf8') as f:
            embed = f.readlines()

        for i in range(len(embed)):
            row_embed = [float(num)
                         for num in embed[i].replace('\n', '').split('\t')[1:]]
            self.emb_dict[name_emb[i]] = row_embed

    def make_emb(self, word):
        try:
            return self.emb_dict_sent[word]
        except KeyError:
            word = [morph.parse(i)[0].normal_form for i in tokenizer.tokenize(re.sub(r'\d+', '', word)) if len(i) > 1]
            emb = [self.emb_dict[i] for i in word]
            return np.mean(emb, axis=0)

    def cos(self, word, label):
        vec1 = self.make_emb(word)
        vec2 = self.make_emb(label)
        return spatial.distance.cosine(vec1, vec2)

    def search_words(self, word):
        for key in self.emb_dict_sent.keys():
            similarity = self.cos(word, key)
            self.answers[key] = similarity
        self.answers = {word: rang for word, rang in sorted(self.answers.items(), key=lambda item: item[1])}
        return self.answers

    def rang(self, word, label):
        words = {word: rang for rang, word in enumerate(self.search_words(word).keys())}
        return words[label]
