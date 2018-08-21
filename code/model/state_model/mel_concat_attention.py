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

import time

def input_concat(args, inputs_list, get_2d=False):

    for i, inputs in enumerate(inputs_list):
        inputs_list[i] = tf.reshape(inputs, [args.batch_size, 1, -1])
    inputs_concat = tf.concat([inputs for inputs in inputs_list], 1)

    if get_2d:
        inputs_concat = tf.reshape(inputs_concat, [args.batch_size, -1])
    
    return inputs_concat

# concatenate 
def concat(inputs, outputs):
    return tf.concat([inputs, outputs], axis=1)

# network
def attention_net(args, inputs, dim_state):
    
    bn = tf.contrib.layers.batch_norm(inputs, is_training=args.is_training)
    fc = tf.contrib.layers.fully_connected(inputs, dim_state,
                        activation_fn=None)

    # feature attention vector = [batch, dim_state]
    #attention = tf.nn.softmax(fc) 
    attention = tf.nn.sigmoid(fc) 
    return attention

# attention model
def attention_model(args, inputs_list, dim_state):
    '''
    input : predicted feature vector of hospital model
    output : feature attention matrix

    feedforward network (MLP)
    '''
    # feature attention net
    attention_list = []
    for inputs in inputs_list:
        attention_list.append(attention_net(args, inputs, dim_state))

    return attention_list

# final fc layer
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
    outputs = tf.contrib.layers.fully_connected(inputs, dim_feature,
                    activation_fn=None)
    #outputs = tf.nn.dropout(outputs, keep_prob=args.keep_prob) 
    return outputs

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
             
            # 1. attention net
            att_A, att_B, att_C = attention_model(self.args,
                                                [self.state_A,
                                                self.state_B,
                                                self.state_C],
                                                dim_state)

            # 2. concatenate state and output of tsl model
            concat_A = concat(self.inputs_A, att_A)
            concat_B = concat(self.inputs_B, att_B)
            concat_C = concat(self.inputs_C, att_C)
           
            # 3. fc layer
            
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
             
