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

import time

def input_concat(args, inputs_list, get_2d=False):

    for i, inputs in enumerate(inputs_list):
        inputs_list[i] = tf.reshape(inputs, [args.batch_size, 1, -1])
    inputs_concat = tf.concat([inputs for inputs in inputs_list], 1)

    if get_2d:
        inputs_concat = tf.reshape(inputs_concat, [args.batch_size, -1])
    
    return inputs_concat

def final_net(args, inputs_list, dim_feature):
    '''
    input : predicted feature vector of hospital model
    output : final predicted feature vector

    feedforward network (MLP)
    '''
    num_tsl = len(inputs_list)

    # merge all of inputs 
    # reshape 3D inputs to 2D
    inputs = input_concat(args, inputs_list, get_2d=True) 

    #bn = tf.contrib.layers.batch_norm(inputs)
    bn = inputs
    outputs = tf.contrib.layers.fully_connected(bn, dim_feature,
                    activation_fn=None)
    #outputs = tf.nn.dropout(outputs, keep_prob=args.keep_prob) 
    return outputs

# Baseline model
class MEL_base_model():

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
            self.inputs_A = tf.placeholder(tf.float32,
                            shape=(args.batch_size, dim_feature))
           
            self.inputs_B = tf.placeholder(tf.float32,
                            shape=(args.batch_size, dim_feature))
            
            self.inputs_C = tf.placeholder(tf.float32,
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

            if args.mode == 'train':
                args.is_training = True
            elif args.mode == 'test':
                args.is_training = False
             
            # 1. final network
            self.predict = final_net(self.args, 
                                [self.state_A,
                                self.state_B,
                                self.state_C],
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
             
