import numpy as np
import tensorflow as tf
import time

def lstm_cell(i,config):
    cell = tf.contrib.rnn.BasicLSTMCell(config.num_hidden, state_is_tuple=True)
    return cell

# LSTM model
class LSTM_Model2():

	def __init__(self, sess, name, config):
		self.sess = sess
		self.name = name
		self._build_net(config)

	def _build_net(self, config):
		# for variables of each models
		with tf.variable_scope(self.name):
			self.training = tf.placeholder(tf.bool)

			# input and target place holders
			self.X = tf.placeholder(tf.float32, [None, config.num_steps, config.dim_inputs])
			self.Y = tf.placeholder(tf.float32, [None, config.dim_targets])
			
			shape = tf.shape(self.X)
			batch_s, seq_length = shape[0], shape[1]
			
			fw_stack_cell = tf.contrib.rnn.MultiRNNCell(
					[lstm_cell(i, config) for i in range(config.num_layers)])
			
			bw_stack_cell = tf.contrib.rnn.MultiRNNCell(
					[lstm_cell(i, config) for i in range(config.num_layers)])
			
			outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_stack_cell, bw_stack_cell, self.X,
			                                                  initial_state_fw=config.fw_state, initial_state_bw=config.bw_state, dtype=tf.float32)
			outputs = tf.concat(outputs, 2)

			self.Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], config.dim_targets,
					activation_fn=None)

		self.loss = tf.reduce_mean(tf.square(self.Y_pred - self.Y))
		optimizer = tf.train.AdamOptimizer(config.learning_rate)
		self.train = optimizer.minimize(self.loss)

		self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.Y_pred - self.Y)))

	def predict(self, feed):
		feed_dict = {self.X: feed[0]}
		return self.sess.run(self.Y_pred, feed_dict=feed_dict)

	def all_rmse(self, feed):
		feed_dict = {self.X: feed[0], self.Y: feed[1]}
		return self.sess.run([self.rmse], feed_dict=feed_dict)

	def learning(self, feed):
		feed_dict = {self.X: feed[0], self.Y: feed[1]}
		return self.sess.run([self.loss, self.train, self.rmse], feed_dict=feed_dict)
