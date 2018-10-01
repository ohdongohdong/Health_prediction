###########################
# Multi Ensemble Learning 
# concat state and output
#
# 1. concatenate state and output of tsl model 
#
# 2. feedforward network
#   - predict the final feature value
#
# input : rnn state by tsl model
# output : predicted next EMR data
#
# by Donghoon Oh
###########################

import os
import numpy as np
import tensorflow as tf
from model_utils import *

import time

# Baseline model
class MEL_concat_base():

    def __init__(self, args, dim_feature, dim_state):
        self.args = args
        self.dim_feature = dim_feature
        self.dim_state = dim_state
        self.build_graph(args, dim_feature, dim_state)

    def build_graph(self, args, dim_feature, dim_state):
        self.graph = tf.Graph()
        with self.graph.as_default():
            
            num_tsl=3
            # input by tsl  = [batch, dim_feature]
            self.value_A = tf.placeholder(tf.float32,
                            shape=(args.batch_size, dim_feature))
           
            self.value_B = tf.placeholder(tf.float32,
                            shape=(args.batch_size, dim_feature))
            
            self.value_C = tf.placeholder(tf.float32,
                            shape=(args.batch_size, dim_feature))
           

            self.state_A = tf.placeholder(tf.float32,
                            shape=(args.batch_size, dim_state))
           
            self.state_B = tf.placeholder(tf.float32,
                            shape=(args.batch_size, dim_state))
            
            self.state_C = tf.placeholder(tf.float32,
                            shape=(args.batch_size, dim_state))
        
            # target = [batch, dim_feature]
            self.targets = tf.placeholder(tf.float32,
                                shape=(args.batch_size, dim_feature))
 
            self.config = {'num_layer': args.num_layer,
                            'hidden_size': args.hidden_size,
                            'learning_rate': args.learning_rate,
                            'keep_prob': args.keep_prob}
 
            # 1. concatenate state and value of tsl model
            concat_A = tf.concat([self.value_A, self.state_A], 1)
            concat_B = tf.concat([self.value_B, self.state_B], 1)
            concat_C = tf.concat([self.value_C, self.state_C], 1)
           
            # 2. fc layer
            self.predict = final_net(self.args, 
                                [concat_A,
                                concat_B,
                                concat_C],
                                dim_feature) 
            
            # decay learning rate
            self.loss = tf.reduce_mean(tf.square(self.predict- self.targets))
            '''
            starter_learning_rate = args.learning_rate
            self.learning_rate = tf.train.exponential_decay(starter_learning_rate, args.epoch,
                                                        args.epoch, 0.96, staircase=True) 
            '''
            self.optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
            self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.predict - self.targets)))
          
            # for counting parameters of model
            self.var_op = tf.global_variables()
            self.var_trainable_op = tf.trainable_variables()
            
            # initialization
            self.initial_op = tf.global_variables_initializer()
            
            # save variables of model
            self.saver = tf.train.Saver(tf.global_variables())
             
