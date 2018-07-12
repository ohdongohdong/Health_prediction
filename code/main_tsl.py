###########################
# Time Series Learning 
# this model is hospital model.
#
# input : previous EMR data
# output : predicted next EMR data
#
# by Donghoon Oh
###########################

import os
import numpy as np
import tensorflow as tf
import time
from data_preprocessing import read_data, padding, split_data

# step 1
# data preprocessing
'''
input : (batch, max_step, input_features)
target : (batch, target_features)
'''

input_set, target_set = read_data('fill')
pad_input_set, seq_len = padding(input_set)

input_train, input_test, target_train, target_test, seq_train, seq_test 
    = split_data(pad_input_set, target_set, seq_len)

# split data set for model

class config(object):
    # data parameters
    num_steps = input_train.shape[1]
    dim_inputs = input_train.shape[2]
    dim_targets = target_train.shape[1]

    # model parameters
    learning_rate = 10e-3
    num_hidden = 512
    num_layers = 3

# hyperparameters
epoch = 1000
batch_size = 256
num_models = 3

with tf.Session() as sess:
    # set ensemble model
    models = []
    for m in range(num_models):
        models.append(LSTM_Model(sess, "model" + str(m), config))

    # Initializate the weights and biases
    tf.global_variables_initializer().run()

    # Training step
    for curr_epoch in range(epoch):
        start = time.time()

        feed1 = [train_inputs1, train_targets1]
        feed2 = [train_inputs2, train_targets2]
        feed = [train_inputs, train_targets]

        train_cost_list = np.zeros(len(models))
        train_rmse_list = np.zeros(len(models))
        for m_idx, m in enumerate(models):
            # if m_idx == 0:
            #     feed = feed1
            # else:
            #     feed = feed2
            train_cost_list[m_idx], _, train_rmse_list[m_idx] = m.learning(feed)

        log = "Epoch {}/{}, train_cost = {}, train_rmse = {}, time = {:.3f}"
        print(log.format(curr_epoch + 1, epoch, train_cost_list, train_rmse_list, time.time() - start))

    print('learning finished')
