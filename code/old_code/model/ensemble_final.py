import numpy as np
import tensorflow as tf
import time

# Prediction fc model
class Pred_Model():

	def __init__(self, sess, name, config):
		self.sess = sess
		self.name = name
		self._build_net(config)

	def _build_net(self, config):
		# for variables of each models
		with tf.variable_scope(self.name):
			self.X = tf.placeholder(tf.float32, [None, config.dim_targets])
			self.Y = tf.placeholder(tf.float32, [None, config.dim_targets])

			self.Y_pred = tf.contrib.layers.fully_connected(self.X, config.dim_targets,
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
