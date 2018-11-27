###########################
# Multi Ensemble Learning 
# Hierarchical model
# 
# 1. feature attention model
#   - create feature attention vector of each of models #       using feature attention network
#   - merge each of vectors to feature attention matrix
#
# 2. hospital attention model
#   - create hospital attention vector by using embeded feature
#       using hospital attention network
#
# 3. feedforward network
#   - predict the final feature value
#       by placing the attention weighted feature as input 
#
# input : predicted next EMR data by tsl model
# output : predicted next EMR data
#
# by Donghoon Oh
###########################

import os
import numpy as np
import tensorflow as tf

import time

# network
def feature_net(args, inputs, dim_feature):
    
    bn = tf.contrib.layers.batch_norm(inputs, is_training=args.is_training)
    fc = tf.contrib.layers.fully_connected(bn, dim_feature,
                        activation_fn=None)

    # feature attention vector = [batch, dim_feature]
    #feat_attention = tf.nn.softmax(fc) 
    feat_attention = tf.nn.sigmoid(fc) 
    return feat_attention

def hospital_net(args, inputs, num_tsl, dim_feature):
     
    # hospital attention vector = [batch, num_tsl]
    bn = tf.contrib.layers.batch_norm(inputs, is_training=args.is_training)
    fc = tf.contrib.layers.fully_connected(bn, num_tsl,
                        activation_fn=None)
            

    #hosp_attention = tf.nn.softmax(fc) 
    hosp_attention = tf.nn.sigmoid(fc) 
    return hosp_attention

def concat(feat_list):
    concatenated = tf.concat([feat for feat in feat_list], 1) 
    return concatenated

# final fc layer
def final_net(args, feat_list, dim_feature, batch_norm=False):
    '''
    output : final predicted feature vector

    feedforward network (MLP)
    '''
    num_tsl = len(feat_list)

    # merge all of inputs 
    inputs = concat(feat_list) 
    print(inputs)

    #fc1 = tf.contrib.layers.fully_connected(inputs, dim_feature,
    #                activation_fn=None)     
    fc1 = tf.contrib.layers.fully_connected(inputs, 256,
                    activation_fn=None)     
    bn = tf.contrib.layers.batch_norm(fc1)
    bn_relu = tf.nn.relu(bn)
    outputs = tf.contrib.layers.fully_connected(bn_relu, dim_feature,
                    activation_fn = None)
    
    #outputs = tf.nn.dropout(outputs, keep_prob=args.keep_prob) 
    return outputs

# attention model
def feat_attention_model(args, inputs_list, dim_feature):
    '''
    input : predicted feature vector of hospital model
    output : feature attention matrix

    feedforward network (MLP)
    '''
    # feature attention net
    beta_list = []
    for inputs in inputs_list:
        beta_list.append(feature_net(args, inputs, dim_feature))

    return beta_list

def hosp_attention_model(args, inputs_A, inputs_B, inputs_C, dim_feature):
    '''
    input : predicted feature vector of hospital model
    output : hospital attention vector

    feedforward network (MLP)
    '''

    num_tsl = 3
    embedding_size = 100
   
    # emdedding feature vectors
    # embeddings = tf.Variable(tf.random_uniform([num_tsl, embedding_size], -1.0, 1.0))
    # embed = tf.nn.embedding_lookup(embeddings, inputs)
    
    # reshape 2D inputs to 3D for concat
    inputs_A = tf.reshape(inputs_A, [args.batch_size, 1, dim_feature])
    inputs_B = tf.reshape(inputs_B, [args.batch_size, 1, dim_feature])
    inputs_C = tf.reshape(inputs_C, [args.batch_size, 1, dim_feature])
     
    # merge all of inputs 
    inputs = tf.concat([inputs_A, inputs_B, inputs_C], 1)

    # reshape 3D inputs to 2D
    inputs = tf.reshape(inputs, [args.batch_size, -1])

    # hospital attention net
    alpha = hospital_net(args, inputs, num_tsl, dim_feature)
    return alpha

# Hierarchical model
class MEL_hierarchical_model():

    def __init__(self, args, dim_feature, dim_state, data_type):
        self.args = args
        self.dim_feature = dim_feature
        self.dim_state = dim_state
        self.data_type = data_type
        self.build_graph(args, dim_feature, dim_state, data_type)

    def build_graph(self, args, dim_feature, dim_state, data_type):
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
           

            # state by tsl  = [batch, dim_state]
            self.state_A = tf.placeholder(tf.float32,
                            shape=(args.batch_size, dim_state))
           
            self.state_B = tf.placeholder(tf.float32,
                            shape=(args.batch_size, dim_state))
            
            self.state_C = tf.placeholder(tf.float32,
                            shape=(args.batch_size, dim_state))
            
         
            # target = [batch, dim_feature]
            self.targets = tf.placeholder(tf.float32,
                                shape=(args.batch_size, dim_feature))
 
            if data_type == 'state':
                inputs_A = self.state_A
                inputs_B = self.state_B
                inputs_C = self.state_C
            elif data_type == 'value':
                inputs_A = self.value_A
                inputs_B = self.value_B
                inputs_C = self.value_C
            
            self.config = {'num_layer': args.num_layer,
                            'hidden_size': args.hidden_size,
                            'learning_rate': args.learning_rate,
                            'keep_prob': args.keep_prob}

            if args.mode == 'train':
                args.is_training = True
            elif args.mode == 'test':
                args.is_training = True
            
            # 1. feature attention model
            # feature attention matrix (beta)
            
            # 1. attention net
            if data_type == 'state':
                att_dim = dim_state
            elif data_type == 'value':
                att_dim = dim_feature
            
            fa_A, fa_B, fa_C = feat_attention_model(self.args,
                                                [inputs_A,
                                                inputs_B,
                                                inputs_C],
                                                att_dim)
           

            # v' = beta * inputs = [batch, dim_feature]
            feat_inputs_A = tf.multiply(fa_A, inputs_A)
            feat_inputs_B = tf.multiply(fa_B, inputs_B)
            feat_inputs_C = tf.multiply(fa_C, inputs_C)
            
            
            # 2. hospital attention model
            # hospital attention vector (alpha)
            ha = hosp_attention_model(self.args,feat_inputs_A,
                                                feat_inputs_B,
                                                feat_inputs_C,
                                                att_dim) 
            
            self.hosp_attention = ha 
            # v'' = alpha * v'
            hosp_inputs_A = tf.multiply(tf.reshape(ha[:,0], [-1,1]), feat_inputs_A)
            hosp_inputs_B = tf.multiply(tf.reshape(ha[:,1], [-1,1]), feat_inputs_B)
            hosp_inputs_C = tf.multiply(tf.reshape(ha[:,2], [-1,1]), feat_inputs_C)
            
            # 3. final network
            self.predict = final_net(self.args, 
                                [hosp_inputs_A,hosp_inputs_B,hosp_inputs_C],
                                dim_feature) 
            
            self.loss = tf.reduce_mean(tf.square(self.predict- self.targets))
            self.optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)

            self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.predict - self.targets)))
          
            # for counting parameters of model
            self.var_op = tf.global_variables()
            self.var_trainable_op = tf.trainable_variables()
            
            # initialization
            self.initial_op = tf.global_variables_initializer()
            
            # save variables of model
            self.saver = tf.train.Saver(tf.global_variables())
             
