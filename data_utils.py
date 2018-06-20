import os
import sklearn
from sklearn.model_selection import StratifiedKFold
import jieba
import pickle
import tensorflow as tf
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold


class DataUtils:
    def __init__(self, batch_size, sentence_len, word2idx, label2idx):
        """

        :param batch_size:
        :param sentence_len:
        :param word2idx:
        :param label2idx:
        """
        self.word2idx = word2idx
        self.label2idx = label2idx
        self.batch_size = batch_size
        self.sentence_len = sentence_len

    def get_train_data(self, file_name, mode="train_"):
        train_dir = os.listdir(file_name)

        all_sentences = []
        all_labels = []

        for v in train_dir:
            if mode in v:
                with open(file_name + v) as fp:
                    tmp = fp.readlines()

                for sentence in tmp:
                    all_sentences.append(sentence.replace("\n", ""))
                    all_labels.append(v.replace(mode, "").replace(".txt", ""))
        return np.array(all_sentences), np.array(all_labels)

    def encoder_data2idx_batch(self, train_data_sentences, train_data_labels):
        encoder_senteces = []
        encoder_labels = []

        for sentence_idx, sentence in enumerate(train_data_sentences):
            tmp = []
            for word_idx, word in enumerate(jieba.lcut(sentence)):
                tmp.append(self.word2idx[word])
            encoder_senteces.append(tmp)
        for v in train_data_labels:
            encoder_labels.append(self.label2idx[v])

        train_X_batches = []
        train_Y_batches = []
        batches_len = []
        all_len = [len(v) for v in encoder_senteces]
        begin_index = 0
        end_index = self.batch_size

        encoder_senteces = tf.keras.preprocessing.sequence.pad_sequences(encoder_senteces, maxlen=self.sentence_len,
                                                                         padding='post')
        encoder_senteces = np.concatenate((encoder_senteces,
                                           np.zeros(
                                               shape=(self.batch_size - len(encoder_senteces) % self.batch_size,
                                                      self.sentence_len),
                                               dtype=np.int32)
                                           ))
        encoder_labels = np.concatenate((encoder_labels,
                                         np.zeros(shape=(self.batch_size - len(encoder_labels) % self.batch_size),
                                                  dtype=np.int32)
                                         ))
        while len(all_len) < len(encoder_senteces):
            all_len += [0]

        while end_index <= len(encoder_senteces):
            train_X_batches.append(encoder_senteces[begin_index:end_index])
            train_Y_batches.append(encoder_labels[begin_index:end_index])
            batches_len.append(all_len[begin_index:end_index])
            begin_index = end_index
            end_index = end_index + self.batch_size

        # bug 未解决
        # c = list(zip(train_X_batches, train_Y_batches))
        # np.random.shuffle(c)
        # train_X_batches[:], train_Y_batches[:] = zip(*c)
        return train_X_batches, train_Y_batches, batches_len

    @staticmethod
    def k_fold(train_sentences, train_labels, k):

        x_train_k_fold = []
        y_train_k_fold = []

        x_test_k_fold = []
        y_test_k_fold = []

        if k == 1:
            return [train_sentences], [train_labels], [], []
        else:
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=2)
            for train_index, test_index in skf.split(train_sentences, train_labels):
                x_train_k_fold.append(train_sentences[train_index])
                y_train_k_fold.append(train_labels[train_index])

                x_test_k_fold.append(train_sentences[test_index])
                y_test_k_fold.append(train_labels[test_index])

        return x_train_k_fold, y_train_k_fold, x_test_k_fold, y_test_k_fold


if __name__ == '__main__':
    word2idx = joblib.load("./dict/word2idx.pkl")
    label2idx = joblib.load("./dict/label_dict.pkl")
    utils = DataUtils(batch_size=60, sentence_len=50, word2idx=word2idx, label2idx=label2idx)
    train_sentences, train_labels = utils.get_train_data("./data/", mode='train_')
    train_X_batches, train_Y_batches, train_len = utils.encoder_data2idx_batch(train_sentences, train_labels)
    x_train_k_fold, y_train_k_fold, x_test_k_fold, y_test_k_fold = DataUtils.k_fold(train_sentences, train_labels, 5)

    develop_sentences, develop_labels = utils.get_train_data("./data/", mode='develop_')
    develop_X_batches, develop_Y_batches, develop_len = utils.encoder_data2idx_batch(train_sentences, train_labels)

    test_sentences, test_labels = utils.get_train_data("./data/", mode='test_')
    test_X_batches, test_Y_batches, test_len = utils.encoder_data2idx_batch(train_sentences, train_labels)
