from scipy import spatial


class Embedding(object):
    def __init__(self):
        self.emb_dict = {}

        with open('./matrix/mentadata_sent.txt', 'r', encoding='utf8') as f:
            name_emb = [row.replace('\n', '') for row in f.readlines()]

        with open('./matrix/final_embeddings.txt', 'r', encoding='utf8') as f:
            embed = f.readlines()

        for i in range(len(embed)):
            row_embed = [float(num)
                         for num in embed[i].replace('\n', '').split('\t')[1:]]
            self.emb_dict[name_emb[i]] = row_embed

    def search_words(self, word):
        answers = {}
        words = {}
        vec1 = self.emb_dict.get(word)
        for key in self.emb_dict.keys():
            vec2 = self.emb_dict.get(key)
            similarity = 1 - spatial.distance.cosine(vec1, vec2)
            if 0 < similarity < 1.0e-02:
                answers[similarity] = key
        for i in sorted(answers.keys()):
            words[answers[i]] = i
        return words
