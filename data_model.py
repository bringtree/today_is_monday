from data_utils import DataUtils


class DataModel:
    def __init__(self, batch_size, fold_num, sentence_len, word2idx, label2idx):
        """

        :param batch_size:
        :param fold_num:
        :param sentence_len:
        :param word2idx:
        :param label2idx:
        """
        k_train_idx_x_batches_list, k_train_y_batches_list, k_train_word_len_batches_list, \
        k_develop_idx_x_batches_list, k_develop_y_batches_list, k_develop_word_len_batches_list, \
        develop_idx_x_batches, develop_y_batches, develop_word_len_batches, \
        test_idx_x_batches, test_y_batches, test_word_len_batches, = \
            self.get_all_data(batch_size=batch_size, fold_num=fold_num, sentence_len=sentence_len, word2idx=word2idx,
                              label2idx=label2idx)
        self.k_train_idx_x_batches_list = k_train_idx_x_batches_list
        self.k_train_y_batches_list = k_train_y_batches_list
        self.k_train_word_len_batches_list = k_train_word_len_batches_list
        self.k_develop_idx_x_batches_list = k_develop_idx_x_batches_list
        self.k_develop_y_batches_list = k_develop_y_batches_list
        self.k_develop_word_len_batches_list = k_develop_word_len_batches_list
        self.develop_idx_x_batches = develop_idx_x_batches
        self.develop_y_batches = develop_y_batches
        self.develop_word_len_batches = develop_word_len_batches
        self.test_idx_x_batches = test_idx_x_batches
        self.test_y_batches = test_y_batches
        self.test_word_len_batches = test_word_len_batches

    def choose_fold(self, fold_idx):
        return self.k_train_idx_x_batches_list[fold_idx], self.k_train_y_batches_list[fold_idx], \
               self.k_train_word_len_batches_list[fold_idx], \
               self.k_develop_idx_x_batches_list[fold_idx], \
               self.k_develop_y_batches_list[fold_idx], self.k_develop_word_len_batches_list[fold_idx]

    def get_develop_data(self):
        return self.develop_idx_x_batches, self.develop_y_batches, self.develop_word_len_batches,

    def get_test_data(self):
        return self.test_idx_x_batches, self.test_y_batches, self.test_word_len_batches,

    @staticmethod
    def get_all_data(batch_size, sentence_len, word2idx, label2idx, fold_num):
        utils = DataUtils(batch_size=batch_size, sentence_len=sentence_len, word2idx=word2idx, label2idx=label2idx)

        # 开发集
        develop_sentences, develop_labels = utils.get_train_data("./data/", mode='develop_')
        develop_idx_x_batches, develop_y_batches, develop_word_len_batches = utils.encoder_data2idx_batch(
            develop_sentences,
            develop_labels)

        # 测试集
        test_sentences, test_labels = utils.get_train_data("./data/", mode='test_')
        test_idx_x_batches, test_y_batches, test_word_len_batches = utils.encoder_data2idx_batch(test_sentences,
                                                                                                 test_labels)
        # 训练集
        train_sentences, train_labels = utils.get_train_data("./data/", mode='train_')
        # 训练集的5折
        k_fold_x_train, k_fold_y_train, k_fold_x_test, k_fold_y_test = DataUtils.k_fold(train_sentences, train_labels,
                                                                                        fold_num)
        # k 代表 训练集切分出来的数据
        k_train_idx_x_batches_list, k_train_y_batches_list, k_train_word_len_batches_list = [], [], []
        k_develop_idx_x_batches_list, k_develop_y_batches_list, k_develop_word_len_batches_list = [], [], []

        for fold_idx in range(fold_num):
            k_train_idx_x_batches, k_train_y_batches, k_train_word_len_batches = utils.encoder_data2idx_batch(
                k_fold_x_train[fold_idx],
                k_fold_y_train[fold_idx])
            k_train_idx_x_batches_list.append(k_train_idx_x_batches)
            k_train_y_batches_list.append(k_train_y_batches)
            k_train_word_len_batches_list.append(k_train_word_len_batches)

            k_develop_idx_x_batches, k_develop_y_batches, k_develop_word_len_batches = utils.encoder_data2idx_batch(
                k_fold_x_test[fold_idx],
                k_fold_y_test[fold_idx])
            k_develop_idx_x_batches_list.append(k_develop_idx_x_batches)
            k_develop_y_batches_list.append(k_develop_y_batches)
            k_develop_word_len_batches_list.append(k_develop_word_len_batches)

        return k_train_idx_x_batches_list, k_train_y_batches_list, k_train_word_len_batches_list, \
               k_develop_idx_x_batches_list, k_develop_y_batches_list, k_develop_word_len_batches_list, \
               develop_idx_x_batches, develop_y_batches, develop_word_len_batches, \
               test_idx_x_batches, test_y_batches, test_word_len_batches,
