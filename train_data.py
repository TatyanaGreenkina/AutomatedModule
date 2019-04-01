import pandas as pd
from nltk.collocations import *
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pymorphy2
from tqdm import tqdm
import re


class DictCreation(object):
    def __init__(self,
                 data_path,
                 columnname,
                 dict_path):
        data_file = pd.read_csv(data_path, sep=';', encoding='cp1251', error_bad_lines=False,
                                low_memory=False)[[columnname]]

        self.morph = pymorphy2.MorphAnalyzer()
        # self.tokenizer = RegexpTokenizer(r'\w+/\w+|\w+-\w+|\w+')
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.rows = data_file.values.reshape((-1))

        words = []

        for row in tqdm(self.rows):
            words.extend([i.lower() for i in self.tokenizer.tokenize(re.sub(r'\d+', ' ', str(row)))
                          if len(i) > 1 and (i == 'то' or i not in stopwords.words('russian'))])

        with open(dict_path, 'w') as f:
            f.write('{}'.format(' '.join(words)))

    def add_bigrams(self):
        # TODO add bigrams to data
        pass
