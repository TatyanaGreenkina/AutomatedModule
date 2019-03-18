from scipy import spatial
import numpy as np
import re
from preprocessing import Preprocessing

prep = Preprocessing(r'./dict', special_token='<UNK>')

class Ranking(object):
    def __init__(self):
        embeddings = []
        with open('./matrix/embeddings.txt', 'r', encoding='utf8') as f:
            emb_lines = f.readlines()
            for i in range(len(emb_lines)):
                embeddings.append([float(num) for num in emb_lines[i].replace('\n', '').split('\t')[1:]])
        embeddings = np.array(embeddings)
        self.final_embeddings = np.concatenate((np.zeros((1, embeddings.shape[1])), embeddings), axis=0)
        self.elems = {i: [prep.word_to_index[j] for j in prep.rows[i].keys() if not re.findall(r'\d', j) and j] for i in
                 prep.rows.keys()}
        self.embed = {key: np.mean(self.final_embeddings[self.elems[key]], axis=0) for key in self.elems.keys()}

    def search_simil_elems(self, elem, num):
        index_words = prep.create_train_data([elem])
        word_emb, embed_dict = self.final_embeddings, self.embed
        search_result = []
        vec1 = np.mean(word_emb[index_words[0]], axis=0)
        for key in embed_dict.keys():
            vec2 = embed_dict[key]
            simil = 1 - spatial.distance.cosine(vec1, vec2)
            if simil > 0.9:
                search_result.append([key, simil])
        search_result.sort(key=lambda x:x[1], reverse=True)
        return search_result[:num]
