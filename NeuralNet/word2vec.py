import re
from collections import Counter, deque
import numpy as np
import pandas as pd
from nltk.collocations import *
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pymorphy2
import tensorflow as tf
import preprocessing

class Word2Vec(object):
    def __init__(self, batch_size, bag_window, embedding_size, vocabulary_size, learning_rate=0.01, save_model=True):
        # ================ Building inputs =================
        self.train_data = tf.placeholder(tf.int32, [batch_size, bag_window * 2])
        self.train_labels = tf.placeholder(tf.int32, [batch_size, 1])
        self.embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), trainable=True)
        # ================== Building the network ==================
        embed = tf.nn.embedding_lookup(self.embeddings, self.train_data)
        context_sum = tf.reduce_sum(embed, 1) / (bag_window * 2)
        score = tf.matmul(context_sum, self.embeddings, transpose_b=True)
        one_hot_labels = tf.one_hot(self.train_labels, depth=vocabulary_size)
        # ================== Loss and train ops ==================
        loss_tensor = tf.nn.softmax_cross_entropy_with_logits_v2(logits=score, labels=one_hot_labels)
        self.loss = tf.reduce_mean(loss_tensor)
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        # ================= Normalize word embeddings for dot product =================
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
        self.normalized_embeddings = self.embeddings / norm
        # ================= Initialize the session =================
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        if save_model:
            self.saver = tf.train.Saver()

    def __call__(self, batch, labels):
        feed_dict = {
            self.train_data: batch,
            self.train_labels: labels
        }
        return self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)

    def create_embeddings(self, save_model=True, model_path=r'./model'):
        if save_model:
            self.saver.save(self.sess, model_path + '/word2vc.ckpt')
        return self.normalized_embeddings.eval(session=self.sess)

    def restore_embeddings(self, model_path):
        self.saver.restore(self.sess, model_path)
        return self.normalized_embeddings.eval(session=self.sess)


def generate_batch(data, data_index, data_size, batch_size, bag_window):
        span = 2 * bag_window + 1  # [ bag_window, target, bag_window ]
        batch = np.ndarray(shape=(batch_size, span - 1), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

        data_buffer = deque(maxlen=span)

        for _ in range(span):
            data_buffer.append(data[data_index])
            data_index = (data_index + 1) % data_size

        for i in range(batch_size):
            data_list = list(data_buffer)
            labels[i, 0] = data_list.pop(bag_window)
            batch[i] = data_list

            data_buffer.append(data[data_index])
            data_index = (data_index + 1) % data_size
        return data_index, batch, labels


#=============================== Experiment ==========================================


# data_file = pd.read_csv('../Data/material.csv', sep=';', encoding = 'cp1251', error_bad_lines=False,
#                         low_memory=False)[['FullName']].astype('str')
#
# morph = pymorphy2.MorphAnalyzer()
# tokenizer = RegexpTokenizer(r'\w+')
#
# words = []
# for row in data_file['FullName'][:1000]:
#     words.extend([morph.parse(i)[0].normal_form.lower() for i in tokenizer.tokenize(re.sub(r'\d+', '', row))
#                   if len(i) > 1 and (i == 'то' or i not in stopwords.words('russian'))])
#
# count = []
# count.extend(Counter(words).most_common())
# word_to_index = {word: i for i, (word, _) in enumerate(count)}
# data = [word_to_index.get(word, 0) for word in words]
# index_to_word = dict(zip(word_to_index.values(), word_to_index.keys()))
#
# num_steps = 2 * len(data)
# loss_every_nsteps = 200
#
# word2vec = Word2Vec(batch_size=128, bag_window=2, embedding_size=64, vocabulary_size=len(count))
# final_embeddings = word2vec.restore_embeddings(model_path=r'../model/word2vc.ckpt')
# print('Initialized')
# average_loss = 0
# for step in range(num_steps):
#     data_index, batch, labels = generate_batch(data_index=0, data_size=len(data), batch_size=128, bag_window=2)
#     _, current_loss = word2vec(batch, labels)
#     average_loss += current_loss
#     if step % loss_every_nsteps == 0:
#         if step > 0:
#             average_loss = average_loss / loss_every_nsteps
#             print("step = {0}, average_loss = {1}".format(step,
#                                                           average_loss))
#             average_loss = 0
# final_embeddings = word2vec.create_embeddings()
# print(final_embeddings[0])
