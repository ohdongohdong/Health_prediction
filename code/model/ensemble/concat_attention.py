###########################
# Multi Ensemble Learning 
# attention using rnn state  of tsl model
# and concat attention and output of tsl
#
# 1. attention network
#   - make attention vector for each of rnn states 
#
# 2. concat attention and output of tsl
#
# 3. feedforward network
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
from model.ensemble.model_utils import *
import time


# Baseline model
class MEL_concat_attention():

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
 
            #weighted = tf.constant([13, 4, 5, 7, 5, 10, 29, 1, 7, 7, 8])
            weighted = tf.constant([0.47, 0.13, 0.16, 0.23, 0.19, 0.34, 1.0, 0.04, 0.24, 0.24, 0.29])
            multiply = tf.constant([args.batch_size])
            self.loss_weight = tf.reshape(tf.tile(weighted, multiply), [multiply[0], dim_feature])
            print(self.loss_weight)
            
            self.config = {'num_layer': args.num_layer,
                            'hidden_size': args.hidden_size,
                            'learning_rate': args.learning_rate,
                            'keep_prob': args.keep_prob}
 
            # 1. attention net
            att_A, att_B, att_C = attention_model(self.args,
                                                [self.state_A,
                                                self.state_B,
                                                self.state_C],
                                                dim_state)

            # 2. concatenate attention by state and value of tsl model
            concat_A = tf.concat([self.value_A, att_A], 1)
            concat_B = tf.concat([self.value_B, att_B], 1)
            concat_C = tf.concat([self.value_C, att_C], 1)
           
            # 3. fc layer 
            self.predict = final_net(self.args, 
                                [concat_A,
                                concat_B,
                                concat_C],
                                dim_feature) 
            
            # decay learning rate
            self.loss = tf.reduce_mean(tf.square(self.predict- self.targets))
            #self.loss = tf.losses.mean_squared_error(self.targets, self.predict, weights=self.loss_weight)
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
             
