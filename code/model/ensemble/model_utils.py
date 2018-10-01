###########################
# Model utile
#
# by Donghoon Oh
###########################

import os
import numpy as np
import tensorflow as tf

def concat(feat_list):
    concatenated = tf.concat([feat for feat in feat_list], 1) 
    return concatenated

def mul(feat_list, att_list):
    mul_list = []
    for feat, att in zip(feat_list, att_list):
        mul_list.append(tf.multiply(att, feat))
    
    return mul_list

# network
def attention_net(args, feature, dim):
    
    fc = tf.contrib.layers.fully_connected(feature, dim,
                        activation_fn=None)

    bn = tf.contrib.layers.batch_norm(fc) 
    
    # attention vector = [batch, dim]
    attention = tf.nn.sigmoid(fc) 
    return attention

# attention model
def attention_model(args, feat_list, dim):
    '''
    input : predicted feature vector of hospital model
    output : feature attention matrix

    feedforward network (MLP)
    '''
    # feature attention net
    attention_list = []
    for feat in feat_list:
        attention_list.append(attention_net(args, feat, dim))
    
    return attention_list

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

    fc1 = tf.contrib.layers.fully_connected(inputs, dim_feature,
                    activation_fn=None)     
    bn = tf.contrib.layers.batch_norm(fc1)
    bn_relu = tf.nn.relu(bn)
    outputs = tf.contrib.layers.fully_connected(bn_relu, dim_feature,
                    activation_fn = None)
    
    #outputs = tf.nn.dropout(outputs, keep_prob=args.keep_prob) 
    return outputs
