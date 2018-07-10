"""
Created on Fri Nov 3 16:07 2017

Prediction health values of next years using RNN

only using train data

@author: Donghoon
"""
import numpy as np
import tensorflow as tf

import time

# Using data that had patient's feature value of all years(2002 ~ 2013)
# load data
data = np.load('../../data/old_data/data.npy')
tag = np.load('../../data/old_data/tag.npy')

# copy from hgkim (I don't know why use only this features)
index_of_inputs = range(3,11)+range(12,15)+range(38,40)	# with weight, height, gender and etc.
index_of_targets = range(2,11)	# features for prediction without persnal information

dim_inputs = len(index_of_inputs)
dim_targets = len(index_of_targets)

# inputs : 2002 ~ 2012, targets : 2013
train_inputs = data[:,:-1,index_of_inputs]
train_targets = data[:,-1,index_of_targets]

print('shape of train inputs: {}'.format(train_inputs.shape))
print('shape of train targets: {}'.format(train_targets.shape))

# data parameters
num_steps = train_inputs.shape[1]   # train data's term of years for prediction

# Training Parameters
learning_rate = 0.0001
epoch = 100
batch_size = None
num_hidden = 300
num_layers = 3
num_train = len(train_inputs)

# input place holders
X = tf.placeholder(tf.float32, [None, num_steps, dim_inputs])
Y = tf.placeholder(tf.float32, [None, dim_targets])

shape = tf.shape(X)
batch_s, seq_length = shape[0], shape[1]

# LSTM model
# build a LSTM network
def lstm_cell():
    return tf.contrib.rnn.BasicLSTMCell(num_hidden)

stack_cell = tf.contrib.rnn.MultiRNNCell(
    [lstm_cell() for _ in range(num_layers)])

outputs, _states = tf.nn.dynamic_rnn(stack_cell, X, dtype=tf.float32)

Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], dim_targets)  # We use the last cell's output

# cost/loss
#loss = tf.reduce_mean(tf.square(Y_pred - Y))  # sum of the squares
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
        train_cost = 0
        train_rmse = 0
        start = time.time()

        feed = {X: train_inputs, Y: train_targets}
        _, train_cost, train_rmse = sess.run([train, loss, rmse], feed_dict=feed)

        train_cost /= num_train
        train_rmse /= num_train
        log = "Epoch {}/{}, train_cost = {:.3f}, train_rmse = {:.3f}, time = {:.3f}"
        print(log.format(curr_epoch + 1, epoch, train_cost, train_rmse, time.time() - start))

    feed = {X: train_inputs, Y: train_targets}
    predic, test_rmse = sess.run([Y_pred, rmse], feed_dict=feed)
    print(test_rmse)
    print(predic[5])
    print(train_targets[5])
