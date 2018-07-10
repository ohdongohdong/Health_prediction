"""
Created on Fri Nov 3 16:07 2017

Prediction health values of next years using LSTM(multi layer)
Add Attention, Dropout

@author: Donghoon
"""

import numpy as np
import tensorflow as tf

import time
import csv
from data_processing import *

# Using data that had patient's feature value of all years(2002 ~ 2013)
# load data
data = np.load('../../data/old_data/data.npy')
tag = np.load('../../data/old_data/tag.npy')

# copy from hgkim (I don't know why use only this features....)
index_of_inputs = range(3,10)+range(12,15)+range(38,40)
index_of_targets = range(5,10) + range(12,15)

# tag of features
tag_inputs = tag[index_of_inputs].tolist()
tag_targets = tag[index_of_targets].tolist()
print(tag_inputs)
print(tag_targets)

# num of features for train and prediction
dim_inputs = len(index_of_inputs)
dim_targets = len(index_of_targets)

# inputs : 2002 ~ 2012, targets : 2013
# split data to train and test by paitents
split = int(len(data)*0.9)

train_inputs = data[:split,:-1,index_of_inputs]
train_targets = data[:split,-1,index_of_targets]

test_inputs = data[split:, :-1, index_of_inputs]
test_targets = data[split:, -1, index_of_targets]

# Check the data format
# inputs = [patients, years(2002 ~ 2012), input_features]
# targets = [patients, year(2013), target_features]
print('shape of train inputs: {}'.format(train_inputs.shape))
print('shape of train targets: {}'.format(train_targets.shape))

print('shape of test inputs: {}'.format(test_inputs.shape))
print('shape of test targets: {}'.format(test_targets.shape))

# data parameters
num_steps = train_inputs.shape[1]   # range of years

# Training Parameters
learning_rate = 0.0001
epoch = 100
batch_size = None
num_hidden = 512
num_layers = 3
num_train = len(train_inputs)

# input place holders
X = tf.placeholder(tf.float32, [None, num_steps, dim_inputs])
Y = tf.placeholder(tf.float32, [None, dim_targets])

shape = tf.shape(X)
batch_s, seq_length = shape[0], shape[1]

# build a LSTM network
# using multi LSTM
def lstm_cell(i):
    cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, state_is_tuple=True)
    #cell = tf.contrib.rnn.GRUCell(num_hidden)
    #cell = tf.contrib.rnn.BasicRNNCell(num_hidden)
    return cell

stack_cell = tf.contrib.rnn.MultiRNNCell(
    [lstm_cell(i) for i in range(num_layers)])

outputs, _states = tf.nn.dynamic_rnn(stack_cell, X, dtype=tf.float32)

# Fully-connected layer
Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], dim_targets)  # We use the last cell's output

# cost/loss
#loss = tf.reduce_mean(tf.square(Y_pred - Y))  # sum of the squares
#loss = tf.sqrt(tf.reduce_mean(tf.square(Y_pred-Y)))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_pred))

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

rmse = tf.sqrt(tf.reduce_mean(tf.square(Y_pred-Y)))

with tf.Session() as sess:
    # Initializate the weights and biases
    tf.global_variables_initializer().run()

    # Training step
    for curr_epoch in range(epoch):
        start = time.time()
	
        feed = {X: train_inputs, Y: train_targets}
        _, train_cost, train_rmse = sess.run([train, loss, rmse], feed_dict=feed)

        log = "Epoch {}/{}, train_cost = {:.3f}, train_rmse = {:.3f}, time = {:.3f}"
        print(log.format(curr_epoch + 1, epoch, train_cost, train_rmse, time.time() - start))

    feed = {X: test_inputs, Y: test_targets}
    predic, test_rmse = sess.run([Y_pred, rmse], feed_dict=feed)


# each RMSE
ex_rmse = RMSE(predic, test_targets)
print(mean_rmse(ex_rmse))
print(predic[0])
print(test_targets[0])
