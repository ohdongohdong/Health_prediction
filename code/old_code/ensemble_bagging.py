"""
Created on Fri Nov 17 16:07 2017

Prediction health values of next years using LSTM

using Multi Bi-directional LSTM

+ Ensemble
using each model for train and test
using a mean of each model's prediction for test
-> bagging

[1. different dataset, same models]
2. different dataset, different models
3. same dataset, different models


@author: Donghoon
"""

import numpy as np
import tensorflow as tf
import time
import csv

from data_processing import *
from ensemble_m1 import LSTM_Model

# load data
data, tag = LoadData()

# Split the data
# inputs : 2002 ~ 2012, targets : 2013
split_point = int(len(data)*0.8)
train_inputs, train_targets, test_inputs, test_targets, tag_inputs, tag_targets = SplitData(data, tag, split_point)

# Print the features for input and target
print('input features : {}'.format(tag_inputs))
print('target features : {}'.format(tag_targets))

# Print the shapes of inputs and targets
print('shape of train inputs: {}'.format(train_inputs.shape))
print('shape of train targets: {}'.format(train_targets.shape))
print('shape of test inputs: {}'.format(test_inputs.shape))
print('shape of test targets: {}'.format(test_targets.shape))


# for ensembel
half = int(len(train_inputs)/2)
train_inputs1 = train_inputs[:half, :, :]
train_inputs2 = train_inputs[half:, :, :]
train_targets1 = train_targets[:half, :]
train_targets2 = train_targets[half:, :]


class config(object):
    # data parameters
    dim_inputs = len(tag_inputs)
    dim_targets = len(tag_targets)
    num_steps = train_inputs.shape[1]  # term of years

    # Training Parameters
    learning_rate = 0.1
    num_hidden = 300
    num_layers = 3
    num_train = len(train_inputs)

# Hyperparameters
epoch = 100
batch_size = None
num_models = 2

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

    # Test each model and check accuracy
    feed = [test_inputs, test_targets]
    predictions = np.zeros([test_targets.shape[0], test_targets.shape[1]])
    test_rmse_list = np.zeros(len(models))
    for m_idx, m in enumerate(models):
        p = m.predict(feed)
        predictions += p
	# avg prediction of two models	
    predic = predictions / num_models

# each RMSE
ex_rmse = RMSE(predic, test_targets)
print(mean_rmse(ex_rmse))

# write each rmse
SaveRMSE(tag_targets, predic, test_targets)
SaveNRMSE(tag_targets, predic, test_targets)

# write each result
SaveReult(tag_targets, predic, test_targets)
