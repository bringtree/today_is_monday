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
import re

params_cfg_file = './params.cfg'
config = configparser.RawConfigParser()
config.read(params_cfg_file)
batch_size_list = [int(v) for v in config.get('hyperparameters', 'batch_size_list').split(",")]
learning_rate_list = [float(v) for v in config.get('hyperparameters', 'learning_rate_list').split(",")]
dropout_list = [float(v) for v in config.get('hyperparameters', 'dropout_list').split(",")]
layer_num_list = [int(v) for v in config.get('hyperparameters', 'layer_num_list').split(",")]
hidden_num_list = [int(v) for v in config.get('hyperparameters', 'hidden_num_list').split(",")]

save_file_num = config.getint('hyperparameters', 'save_file_num')
word_vec_size = config.getint('hyperparameters', 'word_vec_size')
sentence_len = config.getint('hyperparameters', 'sentence_len')
iter_num = config.getint('hyperparameters', 'iter_num')

fold_num = config.getint("plugin", 'fold_num')
base_acc = config.getfloat('plugin', 'base_acc')
base_f1_score = config.getfloat('plugin', 'base_f1_score')
save_pb_mode = config.getboolean('plugin', 'save_pb_mode')
print_bad_case_mode = config.getboolean('plugin', 'print_bad_case_mode')

model_src = config.get('data', 'model_filepath')
idx2vec_path = config.get('data', 'idx2vec_filepath')
word2idx_path = config.get('data', 'word2idx_filepath')
idx2word_path = config.get('data', 'idx2word_filepath')
label2idx_src = config.get('data', 'label2idx_src')

# 模型可视化
# writer = tf.summary.FileWriter("./model_graph/" + visual_model_name)
# writer.add_graph(sess.graph)
# merged_summary = tf.summary.merge_all()
# lstm_model.enable_visual(merged_summary)


# 模型加载和保存
fold_model_src_list = ['./save_model/test/batch_size: 20learning_rate: 0.001dropout: 0.4layer_num: 2hidden_num: 500',
                       './save_model/test/batch_size: 20learning_rate: 0.001dropout: 0.4layer_num: 2hidden_num: 500',
                       './save_model/test/batch_size: 20learning_rate: 0.001dropout: 0.5layer_num: 2hidden_num: 1000',
                       './save_model/test/batch_size: 20learning_rate: 0.001dropout: 0.5layer_num: 3hidden_num: 500',
                       './save_model/test/batch_size: 20learning_rate: 0.001dropout: 0.6layer_num: 2hidden_num: 1000',
                       './save_model/test/batch_size: 20learning_rate: 0.001dropout: 0.6layer_num: 2hidden_num: 2000',
                       './save_model/test/batch_size: 20learning_rate: 0.001dropout: 0.6layer_num: 3hidden_num: 500',
                       ]
layer_num_patten = re.compile('(?<=layer_num: )[0-9]+')
hidden_num_patten = re.compile('(?<=hidden_num: )[0-9]+')

for idx, fold_model_src in enumerate(fold_model_src_list):
    tf.reset_default_graph()
    layer_num = int(re.findall(layer_num_patten, fold_model_src)[0])
    hidden_num = int(int(re.findall(hidden_num_patten, fold_model_src)[0]) / 10)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    lstm_model = model.Model(sentence_len=sentence_len, learning_rate=0.001, word_vec_size=word_vec_size,
                             dropout=1, layer_num=layer_num, hidden_num=hidden_num)
    lstm_model.build(sess)
    sess.run(tf.global_variables_initializer())
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

        with tf.gfile.FastGFile('./pb_model/' + str(idx) + '.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())
