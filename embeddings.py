from scipy import spatial
import re


class Embedding(object):
    def __init__(self):
        self.emb_dict = {}
        self.answers = {}
        with open('./matrix/mentadata_sent.txt', 'r', encoding='utf8') as f:
            name_emb = [row.replace('\n', '') for row in f.readlines()]

        with open('./matrix/final_embeddings.txt', 'r', encoding='utf8') as f:
            embed = f.readlines()

        for i in range(len(embed)):
            row_embed = [float(num)
                         for num in embed[i].replace('\n', '').split('\t')[1:]]
            self.emb_dict[name_emb[i]] = row_embed

    def search_key(self, word):
        word_set = [i for i in re.sub(r'[\W\d]', ' ', word).split(' ') if len(i) > 1]
        for key in self.emb_dict.keys():
            key_set = [i for i in re.sub(r'[\W\d]', ' ', key).split(' ') if len(i) > 1]
            if set(word_set).issubset(set(key_set)):
                return self.emb_dict[key]

    def cos(self, word, label):
        vec1 = self.search_key(word)
        vec2 = self.search_key(label)
        return spatial.distance.cosine(vec1, vec2)

    def search_words(self, word):
        for key in self.emb_dict.keys():
            similarity = self.cos(word, key)
            self.answers[key] = similarity
        self.answers = {word: rang for word, rang in sorted(self.answers.items(), key=lambda item: item[1])}
        return self.answers

    def rang(self, word, label):
        words = {word: rang for rang, word in enumerate(self.search_words(word).keys())}
        return words[label]
