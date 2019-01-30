import numpy as np
import tensorflow as tf
import pandas as pd
from random import shuffle
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import preprocessing
import os
from tqdm import tqdm


class CNN(object):
    def __init__(self,
                 n_tags,
                 emb_mat,
                 n_hidden_list=(128,),
                 cnn_filter_width=7,
                 use_batch_norm=False,
                 embeddings_dropout=False,
                 top_dropout=False,
                 **kwargs):
        # ================ Building inputs =================

        self.learning_rate_ph = tf.placeholder(tf.float32, [])
        self.dropout_keep_ph = tf.placeholder(tf.float32, [])
        self.token_ph = tf.placeholder(tf.int32, [None, None], name='Token_Ind_ph')
        self.y_ph = tf.placeholder(tf.int32, [None], name='y_ph')

        # ================== Building the network ==================

        # Now embedd the indices of tokens using token_emb_dim function
        # this should be like

        emb = tf.nn.embedding_lookup(emb_mat, self.token_ph)
        if embeddings_dropout:
            tf.nn.dropout(emb, self.dropout_keep_ph, (tf.shape(emb)[0], 1, tf.shape(emb)[2]))

        units = emb
        for i in n_hidden_list:
            units = tf.layers.conv1d(units, filters=i, kernel_size=cnn_filter_width, activation=tf.nn.relu,
                                     padding='same')

        units = tf.reduce_max(units, axis=1)

        units = tf.nn.dropout(units, self.dropout_keep_ph, (tf.shape(units)[0], 1))
        logits = tf.layers.dense(units, n_tags, activation=None)
        self.predictions = tf.argmax(logits, 1)

        # ================= Loss and train ops =================
        # Use cross-entropy loss.

        label_hot = tf.one_hot(self.y_ph, depth=n_tags)
        loss_tensor = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_hot, logits=logits)
        self.loss = tf.reduce_mean(loss_tensor)

        # Create a training operation to update the network parameters.
        # We purpose to use the Adam optimizer as it work fine for the
        # most of the cases. Check tf.train to find an implementation.
        # Put the train operation to the attribute self.train_op

        self.train_op = tf.train.AdamOptimizer(self.learning_rate_ph).minimize(self.loss)

        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        if os.listdir(r'./NeuralNet/cnn'):
            self.saver.restore(self.sess, r'./NeuralNet/cnn/cnncnn.ckpt')
        else:
            self.sess.run(tf.global_variables_initializer())

    def __call__(self, tok_batch, mask_batch):
        feed_dict = {self.token_ph: tok_batch,
                     self.dropout_keep_ph: 1.0}
        return self.sess.run(self.predictions, feed_dict)

    def train_on_batch(self, tok_batch, tag_batch, dropout_keep_prob, learning_rate):
        feed_dict = {self.token_ph: tok_batch,
                     self.y_ph: tag_batch,
                     self.dropout_keep_ph: dropout_keep_prob,
                     self.learning_rate_ph: learning_rate}
        self.sess.run(self.train_op, feed_dict)

    def save_model(self, model_path=r'./cnn'):
        self.saver.save(self.sess, model_path + 'cnn.ckpt')

    def predict(self, tok):
        feed_dict = {self.token_ph: tok,
                     self.dropout_keep_ph: 1.0}
        # self.saver.restore(self.sess, r'./NeuralNet/cnn/cnncnn.ckpt')
        return self.sess.run(self.predictions, feed_dict)


def get_emb_mat(emb_mat):
    emb_mat = tf.convert_to_tensor(emb_mat, dtype=tf.float32)
    unk_emb = tf.random.uniform([1, tf.shape(emb_mat)[1]], -1.0, 1.0)
    emb_mat = tf.concat([unk_emb, emb_mat], 0)
    return emb_mat


def get_mask(batch):
    max_len = max(batch, key=len)
    mask = np.zeros((len(batch), len(max_len)), dtype=float)
    index = [[i, j] for (i, row) in enumerate(batch) for (j, _) in enumerate(row)]
    for word in index:
        mask[word[0]][word[1]] = 1
    return mask


def zero_pad(inputs):
    max_len = max(inputs, key=len)
    outputs = []
    for line in inputs:
        while len(line) < len(max_len):
            line.append(0)
        outputs.extend([line])
    return outputs


def prepare_data(file_path, columnname, shuffling=True):
    nomen = pd.read_csv(file_path, sep=';', encoding='cp1251', error_bad_lines=False,
                        low_memory=False)[columnname]
    data = [(elem, label) for (elem, label) in nomen.values]
    vocab = [count for (count, _) in Counter([labels for (_, labels) in data]).most_common()]
    tag_vocab = {count: tag for (tag, count) in enumerate(vocab)}
    if shuffling:
        shuffle(data)
    return data, tag_vocab


def generate_batch(data, data_size, data_index, batch_size, tag_vocab):
    data_bufer = []
    for i in range(batch_size):
        data_bufer.extend([data[data_index]])
        data_index = (data_index + 1) % data_size
    batch = [i for (i, _) in data_bufer]
    label = [i for (_, i) in data_bufer]
    label = [tag_vocab[i] for i in label]
    return batch, label, data_index


def f1_score(n_tag, y_true, y_pred, logs, heatmap=True):
    conf_matrix = np.zeros(shape=(n_tag, n_tag), dtype=int)
    for i in range(len(y_true)):
        conf_matrix[y_true[i]][y_pred[i]] += 1
    precision = np.array([conf_matrix[i]/np.sum(conf_matrix[i]) for i in range(n_tag) if np.sum(conf_matrix[i]) != 0])
    precision = np.mean(precision[np.nonzero(precision)])
    recall = np.array([conf_matrix[:, i]/np.sum(conf_matrix[:, i]) for i in range(n_tag) if np.sum(conf_matrix[:, i]) != 0])
    recall = np.mean(recall[np.nonzero(recall)])
    f1_score = 2*precision*recall/(precision+recall)
    if heatmap:
        df_cm = pd.DataFrame(conf_matrix, index=[i for i in logs], columns=[i for i in logs])
        plt.figure(figsize=(20, 10))
        sns.heatmap(df_cm, annot=True, cmap='YlGnBu', linewidths = .5)
        plt.savefig('./heatmap_1.png')
    return precision, recall, f1_score


def prom_cnn():
    prep = preprocessing.Preprocessing(r'dict', special_token='<UNK>')
    final_embeddings = []
    with open('./matrix/embeddings.txt', 'r', encoding='utf8') as f:
        emb_lines = f.readlines()
        for i in range(len(emb_lines)):
            final_embeddings.append([float(num) for num in emb_lines[i].replace('\n', '').split('\t')[1:]])
    final_embeddings = get_emb_mat(final_embeddings)
    _, tag_vocab = prepare_data(r'./Data/nomenklatura.csv', columnname=['FullName', 'Count'])
    index_to_count = {0: '10,01', 1: '10,02', 2: '10,05', 3: '10,09', 4: '10,06', 5: '10,08', 6: '10,12', 7: '10.10',
                      8: '10.11.01', 9: '10.03', 10: '10.11.02', 11: '10,07', 12: '10,04'}
    cnn = CNN(n_tags=len(tag_vocab), emb_mat=final_embeddings, n_hidden_list=[100, 100, 100])
    # batch = [[prep.word_to_index[i] for i in prep.prepare_data(elem)]]
    # y_pred = index_to_count[cnn.predict(tok=batch)[0]]
    return cnn, prep, index_to_count

#========================================== Experiment ==========================================
# prep = preprocessing.Preprocessing(r'../dict', special_token='<UNK>')
# max_len = prep.length
# # w2v_mat = word2vec.Word2Vec(batch_size=128, bag_window=2, embedding_size=64, vocabulary_size=1744)
# # final_embeddings = w2v_mat.restore_embeddings(model_path=r'../model/word2vc.ckpt')
# final_embeddings = []
# with open('../matrix/embeddings.txt', 'r', encoding='utf8') as f:
#     emb_lines = f.readlines()
#     for i in range(len(emb_lines)):
#         final_embeddings.append([float(num) for num in emb_lines[i].replace('\n', '').split('\t')[1:]])
# final_embeddings = get_emb_mat(final_embeddings)
# _, tag_vocab = prepare_data(r'../Data/nomenklatura.csv', columnname=['FullName', 'Count'])
# print(tag_vocab)
# data, _ = prepare_data(r'../Data/handysoft.csv', columnname=['Номенклатура', 'Счет'])
# data_valid = []
# for elem, label in data:
#     if label == '10.июн':
#         data_valid.append((elem, '10,06'))
#     if label == '10.сен':
#         data_valid.append((elem, '10,09'))
#     if label == '10.янв':
#         data_valid.append((elem, '10,01'))
#     if label == '10.май':
#         data_valid.append((elem, '10,05'))
#
#
# batch_size = 32
# n_epochs = 1
# learning_rate = 0.001
# dropout_keep_prob = 0.7
#
# cnn = CNN(n_tags=len(tag_vocab), emb_mat=final_embeddings, n_hidden_list=[100, 100, 100])
#
# num_steps = n_epochs * (len(data_valid) // batch_size)
# data_index = 0
# total_pred = []
# total_true = []
# for step in tqdm(range(num_steps)):
#     batch, label, data_index = generate_batch(data_valid, len(data_valid), data_index, batch_size, tag_vocab)
#     batch = prep.create_train_data(batch)
#     x_batch = zero_pad(batch)
#     # cnn.train_on_batch(x_batch, label, dropout_keep_prob, learning_rate)
#     y_pred = cnn.predict(tok=x_batch)
#     total_pred.extend(y_pred)
#     total_true.extend(label)
# # cnn.save_model()
# logs = ['10,01', '10,02-нет', '10,05', '10,09', '10,06', '10,08-нет', '10,1-нет', '10,10-нет', '10,11,01-нет', '10,03',
#         '10,11,02-нет', '10,07-нет', '10,04-нет']
# precision, recall, f1_score = f1_score(len(tag_vocab), total_true, total_pred, logs=logs)
# print('Precision: {}, recall: {}, f1_score: {}'.format(precision, recall, f1_score))

#======================================================================================================================

# nomen = pd.read_csv('../Data/nomenklatura.csv', sep=';', encoding='cp1251', error_bad_lines=False, low_memory=False)[['FullName', 'Count']]

# x = tf.random_normal(shape=[13, 4, 3])
# x_emb = tf.random.uniform(shape=[3, 4], minval=1, maxval=10, dtype=tf.int32)
# y = tf.layers.conv1d(x, filters=2, kernel_size=2, padding='same')
# pooling = tf.reduce_max(y, axis=1)
# z = tf.nn.dropout(pooling, 0.7, (tf.shape(pooling)[0], 1))
# logits = tf.layers.dense(z, 13, activation=None)
# predictions = tf.argmax(logits, 1)
# mean = tf.reduce_mean(logits)
# label = tf.random_uniform(shape=[13], minval=1, maxval=13, dtype=tf.int32)
# label_hot = tf.one_hot(label, depth=13)
# loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_hot, logits=logits)
# emb = tf.nn.embedding_lookup(final_embeddings[:15], x_emb)

#
# def print_shape(out1):
#     print(out1)
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         print(sess.run(out1))
#
#
# print_shape(logits)

# y_true = [3,  1,  1,  5, 10,  4, 12,  2,  1,  7, 3,  1,  1,  5, 10,  4, 12,  2,  1,  7]
# y_pred = [3,  4,  1,  5, 12,  4, 1,  1,  1,  7, 3,  1,  1,  5, 10,  4, 12,  2,  1,  7]
# #
# conf_matrix = np.zeros(shape=(13, 13), dtype=int)
#
# for i in range(len(y_true)):
#     conf_matrix[y_true[i]][y_pred[i]] += 1

# precision = np.array([conf_matrix[i]/np.sum(conf_matrix[i]) for i in range(13) if np.sum(conf_matrix[i]) != 0])
# precision = np.mean(precision[np.nonzero(precision)])
# recall = np.array([conf_matrix[:, i]/np.sum(conf_matrix[:, i]) for i in range(13) if np.sum(conf_matrix[:, i]) != 0])
# recall = np.mean(recall[np.nonzero(recall)])
# f1_score = 2*precision*recall/(precision+recall)
# df_cm = pd.DataFrame(conf_matrix, index= [i for i in rows], columns=[i for i in rows])
# plt.figure()
# sns.heatmap(df_cm, annot=True, cmap='YlGnBu')
# plt.savefig('../heatmap.png')
#
#
# print(precision, recall, f1_score)
