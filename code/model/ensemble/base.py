###########################
# Multi Ensemble Learning 
# Baseline model
# 
# 1. feedforward network
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

# Baseline model
class MEL_base_model():

    def __init__(self, args, dim_feature, dim_state, data_type):
        self.args = args
        self.dim_feature = dim_feature
        self.dim_state = dim_state
        self.data_type = data_type
        self.build_graph(args, dim_feature, dim_state, data_type)

    def build_graph(self, args, dim_feature, dim_state, data_type):
        self.graph = tf.Graph()
        with self.graph.as_default():
            
            # input by tsl  = [batch, dim_feature]
            self.value_A = tf.placeholder(tf.float32,
                            shape=(args.batch_size, dim_feature))
           
            self.value_B = tf.placeholder(tf.float32,
                            shape=(args.batch_size, dim_feature))
            
            self.value_C = tf.placeholder(tf.float32,
                            shape=(args.batch_size, dim_feature))
           

            # state by tsl  = [batch, dim_state]
            self.state_A = tf.placeholder(tf.float32,
                            shape=(args.batch_size, dim_state))
           
            self.state_B = tf.placeholder(tf.float32,
                            shape=(args.batch_size, dim_state))
            
            self.state_C = tf.placeholder(tf.float32,
                            shape=(args.batch_size, dim_state))
       
            if data_type == 'state':
                inputs_A = self.state_A
                inputs_B = self.state_B
                inputs_C = self.state_C
            elif data_type == 'value':
                inputs_A = self.value_A
                inputs_B = self.value_B
                inputs_C = self.value_C

            # target = [batch, dim_feature]
            self.targets = tf.placeholder(tf.float32,
                                shape=(args.batch_size, dim_feature))
 
            self.config = {'num_layer': args.num_layer,
                            'hidden_size': args.hidden_size,
                            'learning_rate': args.learning_rate,
                            'keep_prob': args.keep_prob}

            if args.mode == 'train':
                args.is_training = True
            elif args.mode == 'test':
                args.is_training = False
             
            # 1. final network
            self.predict = final_net(self.args, 
                                [inputs_A,
                                 inputs_B,
                                 inputs_C],
                                dim_feature,
                                 batch_norm=False)
           
            # train loss
            self.loss = tf.reduce_mean(tf.square(self.predict- self.targets))
            '''
            starter_learning_rate = args.learning_rate
            self.learning_rate = tf.train.exponential_decay(starter_learning_rate, args.epoch,
                                                        args.epoch, 0.96, staircase=True) 
            '''
            self.optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
            # RMSE 
            self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.predict - self.targets)))
          
            # for counting parameters of model
            self.var_op = tf.global_variables()
            self.var_trainable_op = tf.trainable_variables()
            
            # initialization
            self.initial_op = tf.global_variables_initializer()
            
            # save variables of model
            self.saver = tf.train.Saver(tf.global_variables())
             
