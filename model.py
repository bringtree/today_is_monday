import tensorflow as tf
import numpy as np
import focal_loss


class Model():
    """
    lstm 模型
    """

    def __init__(self, sentence_len=50, learning_rate=0.005, word_vec_size=400, hidden_num=300,
                 dropout=0.5, layer_num=2):
        """

        :param sentence_len: 句子长度
        :param learning_rate: 学习速率
        :param word_vec_size: 词向量的大小
        :param hidden_num: lstm 隐藏层 size 大小
        :param batch_size: batch_size 的大小
        """
        self.sentence_len = sentence_len
        self.learning_rate = learning_rate
        self.word_vec_size = word_vec_size
        self.hidden_num = hidden_num
        self.dropout = dropout
        self.batch_size = None
        self.layer_num = layer_num
        self.sess = None
        self.merged_summary = None

    def build(self, sess):
        """
        构建 lstm 模型
        :return: Model对象
        """
        self.sess = sess
        self.input_sentences = tf.placeholder(shape=[None, self.sentence_len, self.word_vec_size],
                                              name='input_sentences',
                                              dtype=tf.float32)
        self.input_labels = tf.placeholder(shape=[None], name='input_labels',
                                           dtype=tf.int32)
        sentence_input = tf.transpose(self.input_sentences, [1, 0, 2])

        self.input_length = tf.placeholder(shape=[None], name='input_length', dtype=tf.int32)
        self.dropout_rate = tf.placeholder(shape=None, name='dropout_rate', dtype=tf.float32)

        with tf.variable_scope("bid_lstm_layer"):
            def get_drop_lstm():
                return tf.contrib.rnn.DropoutWrapper(
                    tf.contrib.rnn.LSTMCell(self.hidden_num, initializer=tf.orthogonal_initializer()),
                    input_keep_prob=self.dropout_rate,
                    output_keep_prob=self.dropout_rate)

            # forward_lstm = tf.contrib.rnn.LSTMCell(self.hidden_num, initializer=tf.orthogonal_initializer())
            # backward_lstm = tf.contrib.rnn.LSTMCell(self.hidden_num, initializer=tf.orthogonal_initializer())

            # forward_drop = tf.contrib.rnn.DropoutWrapper(forward_lstm, input_keep_prob=self.dropout_rate,
            #                                              output_keep_prob=self.dropout_rate)

            # backward_drop = tf.contrib.rnn.DropoutWrapper(backward_lstm, input_keep_prob=self.dropout_rate,
            #                                               output_keep_prob=self.dropout_rate)

            forward_drop_mul = tf.nn.rnn_cell.MultiRNNCell([get_drop_lstm() for _ in range(self.layer_num)],
                                                           state_is_tuple=True)
            backward_drop_mul = tf.nn.rnn_cell.MultiRNNCell([get_drop_lstm() for _ in range(self.layer_num)],
                                                            state_is_tuple=True)

            # shape=(50, ?, 300) dtype=float32>
            encoder_outputs, encoder_final_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=forward_drop_mul,
                cell_bw=backward_drop_mul,
                inputs=sentence_input,
                sequence_length=self.input_length,
                dtype=tf.float32,
                time_major=True)
            # 300 + 300 + 300 + 300
            flatten_encoder_final_state = []
            for v in encoder_final_state:
                for j in v:
                    flatten_encoder_final_state.append(j.h)

            lstm_output = tf.concat(flatten_encoder_final_state, 1)

        with tf.variable_scope('full_connection_layer'):
            intent_w = tf.get_variable(
                initializer=tf.random_uniform([self.hidden_num * 2 * self.layer_num, 2], -0.1, 0.1),
                dtype=tf.float32, name="intent_w")
            intent_b = tf.get_variable(initializer=tf.ones([2]), dtype=tf.float32, name="intent_b")
            predict_intent = tf.matmul(lstm_output, intent_w) + intent_b

        with tf.variable_scope('predict_result'):
            self.output_intent = tf.argmax(predict_intent, 1, name='output_result')

        with tf.variable_scope("loss_function"):
            cross_entropy = focal_loss.focal_loss(
                prediction_tensor=predict_intent, target_tensor=tf.one_hot(self.input_labels, 2), weights=None,
                alpha=0.5,
                gamma=2
            )
            self.loss = tf.reduce_mean(cross_entropy)

        with tf.variable_scope("optimizer_function"):
            # all_vars = tf.trainable_variables()
            # optimizer = tf.keras.optimizers.SGD(lr=self.learning_rate, decay=0.0005).get_updates(self.loss,
            #                                                                                      params=all_vars)
            # self.train_op = optimizer
            #
            optimizer = tf.train.AdamOptimizer(name="a_optimizer", learning_rate=self.learning_rate)
            self.grads, self.vars = zip(*optimizer.compute_gradients(self.loss))
            self.gradients, _ = tf.clip_by_global_norm(self.grads, 5)  # clip gradients
            self.train_op = optimizer.apply_gradients(zip(self.gradients, self.vars))

        with tf.variable_scope("visual_score"):

            self.develop_accuracy_placeholder = tf.placeholder(shape=None, dtype=tf.float32,
                                                               name="develop_accuracy_placeholder")
            self.test_accuracy_placeholder = tf.placeholder(shape=None, dtype=tf.float32,
                                                            name="test_accuracy_placeholder")
            self.k_train_accuracy_placeholder = tf.placeholder(shape=None, dtype=tf.float32,
                                                               name="k_train_accuracy_placeholder")
            self.k_develop_accuracy_placeholder = tf.placeholder(shape=None, dtype=tf.float32,
                                                                 name="k_develop_accuracy_placeholder")

            self.develop_f1_placeholder = tf.placeholder(shape=None, dtype=tf.float32,
                                                         name="develop_f1_placeholder")
            self.test_f1_placeholder = tf.placeholder(shape=None, dtype=tf.float32,
                                                      name="test_f1_placeholder")
            self.k_train_f1_placeholder = tf.placeholder(shape=None, dtype=tf.float32,
                                                         name="k_train_f1_placeholder")
            self.k_develop_f1_placeholder = tf.placeholder(shape=None, dtype=tf.float32,
                                                           name="k_develop_f1_placeholder")

            self.develop_accuracy_op = self.develop_accuracy_placeholder
            self.test_accuracy_op = self.test_accuracy_placeholder
            self.k_train_accuracy_op = self.k_train_accuracy_placeholder
            self.k_develop_accuracy_op = self.k_develop_accuracy_placeholder

            self.develop_f1_op = self.develop_f1_placeholder
            self.test_f1_op = self.test_f1_placeholder
            self.k_train_f1_op = self.k_train_f1_placeholder
            self.k_develop_f1_op = self.k_develop_f1_placeholder

            tf.summary.scalar("develop_accuracy", self.develop_accuracy_placeholder)
            tf.summary.scalar("test_accuracy", self.test_accuracy_placeholder)
            tf.summary.scalar("k_train_accuracy", self.k_train_accuracy_placeholder)
            # tf.summary.scalar("k_develop_accuracy", self.k_develop_accuracy_placeholder)

            tf.summary.scalar("develop_f1", self.develop_f1_placeholder)
            tf.summary.scalar("test_f1", self.test_f1_placeholder)
            tf.summary.scalar("k_train_f1", self.k_train_f1_placeholder)
            # tf.summary.scalar("k_develop_f1", self.k_develop_f1_placeholder)

    def enable_visual(self, merged_summary):
        self.merged_summary = merged_summary

    def train(self, input_sentences, input_labels, input_length):
        """

        :param input_sentences: 训练的句子 shape = [batch_size, self.sentence_len, self.word_vec_size],
        :param input_labels: 训练的标签 shape = [batch_size]
        :param input_length: 训练的句子长度 shape = [batch_size]
        :return: loss大小 int
        """
        loss, result, _ = self.sess.run([self.loss, self.output_intent, self.train_op], feed_dict={
            self.input_sentences: input_sentences,
            self.input_labels: input_labels,
            self.input_length: input_length,
            self.dropout_rate: self.dropout
        })
        return loss, result

    def predict(self, input_sentences, input_length):
        """

        :param input_sentences: 预测的句子 shape = [batch_size, self.sentence_len, self.wordVec_size],
        :param input_length: 预测的标签 shape = [batch_size]
        :return: 预测的结果 shape = [batch_size]
        """
        result = self.sess.run([self.output_intent], feed_dict={
            self.input_sentences: input_sentences,
            self.input_length: input_length,
            self.dropout_rate: 1
        })
        return result[0]

    def get_predict_score(self, x_batches, y_batches, word_len_batches, word2idx, idx2vec, idx2word,
                          train_mode,
                          print_bad_case_mode=False):
        loss, right_num, err_num, TP, FP, TN, FN, = 0, 0, 0, 0, 0, 0, 0
        for i in range(len(x_batches)):
            x = x_batches[i]
            y = y_batches[i]
            word_len = word_len_batches[i]
            encoder_x = []

            for sentence in x:
                tmp = []
                for word_idx in sentence:
                    if word_idx in idx2vec:
                        tmp.append(idx2vec[word_idx])
                    else:
                        if word2idx is 0:
                            tmp.append(np.zeros(self.word_vec_size, dtype=np.float32))
                        else:
                            tmp.append(np.ones(self.word_vec_size, dtype=np.float32))
                encoder_x.append(np.array(tmp))

            if train_mode is True:
                tmp_loss, result = self.train(
                    input_sentences=encoder_x,
                    input_labels=y,
                    input_length=word_len
                )
                loss += tmp_loss
            else:
                result = self.predict(
                    input_sentences=encoder_x,
                    input_length=word_len
                )
                # 计算f1 分数
            for idx, v in enumerate([v for v in word_len if v != 0]):
                if result[idx] == 0:
                    if y[idx] == result[idx]:
                        TP += 1
                        right_num += 1
                    else:
                        FP += 1
                        err_num += 1
                        if print_bad_case_mode is True:
                            sentence_tmp = []
                            for word_idx in x[idx]:
                                if word_idx == 0:
                                    pass
                                elif word_idx in idx2word:
                                    sentence_tmp.append(idx2word[word_idx])
                                else:
                                    sentence_tmp.append('unknown_word')
                            print(''.join(sentence_tmp) + " true intent:" + str(y[idx]) + " predict intent:" + str(
                                result[idx]))
                else:
                    if y[idx] == result[idx]:
                        TN += 1
                        right_num += 1
                    else:
                        FN += 1
                        err_num += 1
                        if print_bad_case_mode is True:
                            sentence_tmp = []
                            for word_idx in x[idx]:
                                if word_idx == 0:
                                    pass
                                elif word_idx in idx2word:
                                    sentence_tmp.append(idx2word[word_idx])
                                else:
                                    sentence_tmp.append('unknown_word')
                            print(''.join(sentence_tmp))
                            print("true intent:" + str(y[idx]) + " predict intent:" + str(result[idx]))

        precision = TP / (TP + FP + 1e-5)
        recall = TP / (TP + FN + 1e-5)
        acc = right_num / (err_num + right_num)
        # print("precision %.6f recall %.6f" % (precision, recall))
        f1_score = 2 * precision * recall / (precision + recall + 1e-5)
        if train_mode is True:
            return acc, f1_score, loss
        else:
            return acc, f1_score

    def visual_result(self,
                      develop_accuracy,
                      test_accuracy,
                      k_develop_accuracy,
                      k_train_accuracy,
                      develop_f1,
                      test_f1,
                      k_develop_f1,
                      k_train_f1, ):
        if self.merged_summary is None:
            raise Exception("check the merged_summary mode, it is None ")
        if (k_develop_accuracy is None) and (k_develop_f1 is None):
            result_board = self.sess.run([
                self.develop_accuracy_op,
                self.test_accuracy_op,
                self.k_train_accuracy_op,
                self.develop_f1_op,
                self.test_f1_op,
                self.k_train_f1_op,
                self.merged_summary],
                feed_dict={
                    self.develop_accuracy_placeholder: develop_accuracy,
                    self.test_accuracy_placeholder: test_accuracy,
                    self.develop_f1_placeholder: develop_f1,
                    self.test_f1_placeholder: test_f1,
                    self.k_train_accuracy_placeholder: k_train_accuracy,
                    self.k_train_f1_placeholder: k_train_f1,
                })
        else:
            result_board = self.sess.run([
                self.develop_accuracy_op,
                self.test_accuracy_op,
                self.k_train_accuracy_op,
                self.k_develop_accuracy_op,
                self.develop_f1_op,
                self.test_f1_op,
                self.k_train_f1_op,
                self.k_develop_f1_op,
                self.merged_summary],
                feed_dict={
                    self.develop_accuracy_placeholder: develop_accuracy,
                    self.test_accuracy_placeholder: test_accuracy,
                    self.k_train_accuracy_placeholder: k_train_accuracy,
                    self.k_develop_accuracy_placeholder: k_develop_accuracy,
                    self.develop_f1_placeholder: develop_f1,
                    self.test_f1_placeholder: test_f1,
                    self.k_train_f1_placeholder: k_train_f1,
                    self.k_develop_f1_placeholder: k_develop_f1,
                })
        return result_board[-1]


# 测试
if __name__ == '__main__':
    sess = tf.Session()
    lstm_model = Model(sentence_len=50, learning_rate=1, word_vec_size=2, hidden_num=100,
                       )
    lstm_model.build(sess)
    sess.run(tf.global_variables_initializer())
    s = lstm_model.train(
        input_sentences=[[[v * 0.1, 1] for v in range(50)],
                         [[v * 0.2, 1] for v in range(50)],
                         [[v * 0.3, 1] for v in range(50)]
                         ],
        input_labels=[0, 1, 0],
        input_length=[3, 2, 1]
    )
    print(s)
