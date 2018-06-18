# PYTHONUNBUFFERED=1;LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64
import sys
import os
import tensorflow as tf
import model
import joblib
import numpy as np
import jieba
from tensorflow.python.framework import graph_util
import configparser
from data_utils import DataUtils
import multiprocessing
from multiprocessing import Process, Manager
from sklearn.model_selection import StratifiedKFold
from data_model import DataModel


def cuda_run(data_model, learning_rate, dropout, layer_num, hidden_num, k_idx,
             save_file_num, word_vec_size, sentence_len, iter_num, base_acc, base_f1_score, save_pb_mode,
             print_bad_case_mode,
             pb_file, model_src, idx2vec, word2idx, idx2word, ):
    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    lstm_model = model.Model(sentence_len=sentence_len, learning_rate=learning_rate, word_vec_size=word_vec_size,
                             dropout=dropout, layer_num=layer_num, hidden_num=hidden_num)
    lstm_model.build(sess)
    sess.run(tf.global_variables_initializer())

    # 模型可视化
    writer = tf.summary.FileWriter("./model_graph/test" + str(k_idx))
    writer.add_graph(sess.graph)
    merged_summary = tf.summary.merge_all()
    lstm_model.enable_visual(merged_summary)

    # 模型加载和保存
    fold_model_src = model_src + str(k_idx)
    saver = tf.train.Saver(max_to_keep=int(save_file_num))
    ckpt = tf.train.get_checkpoint_state(fold_model_src)

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    if save_pb_mode is True:
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            ['predict_result/output_result'],
            variable_names_whitelist=None,
            variable_names_blacklist=None
        )
        with tf.gfile.FastGFile('./' + pb_file, mode='wb') as f:
            f.write(constant_graph.SerializeToString())

    k_train_idx_x_batches, k_train_y_batches, k_train_word_len_batches, \
    k_develop_idx_x_batches, k_develop_y_batches, k_develop_word_len_batches = \
        data_model.choose_fold(fold_idx=k_idx)
    develop_idx_x_batches, develop_y_batches, develop_word_len_batches = data_model.get_develop_data()
    test_idx_x_batches, test_y_batches, test_word_len_batches = data_model.get_test_data()
    for iter in range(iter_num):
        develop_acc, develop_f1_score = lstm_model.get_predict_score(develop_idx_x_batches,
                                                                     develop_y_batches,
                                                                     develop_word_len_batches, word2idx, idx2vec,
                                                                     idx2word,
                                                                     False,
                                                                     print_bad_case_mode)
        test_acc, test_f1_score = lstm_model.get_predict_score(test_idx_x_batches, test_y_batches,
                                                               test_word_len_batches, word2idx, idx2vec, idx2word,
                                                               False,
                                                               print_bad_case_mode)
        k_develop_acc, k_develop_f1_score = lstm_model.get_predict_score(k_develop_idx_x_batches,
                                                                         k_develop_y_batches,
                                                                         k_train_word_len_batches, word2idx,
                                                                         idx2vec,
                                                                         idx2word,
                                                                         False,
                                                                         print_bad_case_mode)
        k_train_acc, k_train_f1_score, k_loss = lstm_model.get_predict_score(k_train_idx_x_batches,
                                                                             k_train_y_batches,
                                                                             k_train_word_len_batches, word2idx,
                                                                             idx2vec,
                                                                             idx2word,
                                                                             True,
                                                                             False)

        result_board = lstm_model.visual_result(develop_accuracy=develop_acc,
                                                test_accuracy=test_acc,
                                                k_test_accuracy=k_train_acc,
                                                develop_f1=develop_f1_score,
                                                test_f1=k_develop_acc,
                                                k_test_f1=k_train_f1_score)
        writer.add_summary(result_board, iter)

        # 保存模型
        if (k_train_acc > base_acc) and (k_train_f1_score > base_f1_score):
            if (k_develop_acc > base_acc) and (k_develop_f1_score > base_f1_score):
                if (develop_acc > base_acc) and (develop_f1_score > base_f1_score):
                    if (test_acc > base_acc) and (test_f1_score > base_f1_score):
                        if not os.path.exists(fold_model_src):
                            os.makedirs(fold_model_src)
                        print("saving model " + str(fold_model_src))
                        saver.save(sess, fold_model_src, global_step=iter)
                        return


if __name__ == '__main__':
    params_cfg_file = './params.cfg'
    config = configparser.RawConfigParser()
    config.read(params_cfg_file)
    batch_size_list = [int(v) for v in config.get('hyperparameters', 'batch_size').split(",")]
    learning_rate_list = [float(v) for v in config.get('hyperparameters', 'learning_rate').split(",")]
    dropout_list = [float(v) for v in config.get('hyperparameters', 'dropout').split(",")]
    layer_num_list = [int(v) for v in config.get('hyperparameters', 'layer_num').split(",")]
    hidden_num_list = [int(v) for v in config.get('hyperparameters', 'hidden_num').split(",")]

    save_file_num = config.getint('hyperparameters', 'save_file_num')
    word_vec_size = config.getint('hyperparameters', 'word_vec_size')
    sentence_len = config.getint('hyperparameters', 'sentence_len')
    iter_num = config.getint('hyperparameters', 'iter_num')

    fold_num = config.getint("plugin", 'fold_num')
    base_acc = config.getfloat('plugin', 'base_acc')
    base_f1_score = config.getfloat('plugin', 'base_f1_score')
    save_pb_mode = config.getboolean('plugin', 'save_pb_mode')
    print_bad_case_mode = config.getboolean('plugin', 'print_bad_case_mode')

    pb_file = config.get('data', 'pb_filepath')
    model_src = config.get('data', 'model_filepath')
    idx2vec_path = config.get('data', 'idx_vec_filepath')
    word2idx_path = config.get('data', 'word2idx_filepath')
    idx2word_path = config.get('data', 'idx2word_filepath')
    label2idx_src = config.get('data', 'label2idx_src')

    idx2vec = joblib.load(idx2vec_path)
    word2idx = joblib.load(word2idx_path)
    idx2word = joblib.load(idx2word_path)
    label2idx = joblib.load(label2idx_src)
    for model_params_idx in range(1):
        process_list = []
        batch_size = batch_size_list[model_params_idx]
        learning_rate = learning_rate_list[model_params_idx]
        dropout = dropout_list[model_params_idx]
        layer_num = layer_num_list[model_params_idx]
        hidden_num = hidden_num_list[model_params_idx]

        data_model = DataModel(batch_size=batch_size, fold_num=fold_num, sentence_len=sentence_len, word2idx=word2idx,
                               label2idx=label2idx)
        for k_idx in range(fold_num):
            process_list.append(
                multiprocessing.Process(target=cuda_run,
                                        args=(
                                            data_model, learning_rate, dropout, layer_num, hidden_num, k_idx,
                                            save_file_num, word_vec_size, sentence_len, iter_num, base_acc,
                                            base_f1_score,
                                            save_pb_mode,
                                            print_bad_case_mode,
                                            pb_file, model_src, idx2vec, word2idx, idx2word,)))

        for process in process_list:
            process.start()
        for process in process_list:
            process.join()
