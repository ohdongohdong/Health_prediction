###########################
# Time Series Learning 
# this model is hospital model.
#
# input : previous EMR data
# output : predicted next EMR data
#
# by Donghoon Oh
###########################

import os
import numpy as np
import tensorflow as tf

import time

def rnn_cell(args):
    if args.model == 'A':
        cell = tf.contrib.rnn.BasicLSTMCell(args.hidden_size, state_is_tuple=True) 
#        cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
#                                    args.hidden_size,
#                                    dropout_keep_prob=args.keep_prob)
    elif args.model == 'B':
        cell = tf.contrib.rnn.BasicRNNCell(args.hidden_size)
    elif args.model == 'C':
        cell = tf.contrib.rnn.UGRNNCell(args.hidden_size)
    else:
        raise TypeError('model should be A, B or C.')
    return cell

def build_BRNN(args, input, seq_len):
    
    output = input
    for i in range(args.num_layer):
        scope = 'BRNN_' + str(i+1)
        fw_cell = rnn_cell(args)
        bw_cell = rnn_cell(args)

        _initial_state_fw = fw_cell.zero_state(args.batch_size, tf.float32)
        _initial_state_bw = bw_cell.zero_state(args.batch_size, tf.float32)

        # tensor = [batch_size, time_step, input_feature]
        outputs, output_states =\
                        tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                                                        inputs=output,
                                                        sequence_length=seq_len,
                                                        initial_state_fw = _initial_state_fw,
                                                        initial_state_bw = _initial_state_bw,
                                                        scope=scope)
        
        output_fw, output_bw = outputs 

        if i == args.num_layer-1:
            output_fw = tf.transpose(output_fw, [1,0,2])
            output_bw = tf.transpose(output_bw, [1,0,2])
            output = tf.concat([output_fw[-1], output_bw[-1]], axis=1)
            print(output_states)
            output_states = tf.concat([output_states[0], output_states[1]],axis=1) 
        else:
            output = tf.concat([output_fw, output_bw], 2)
    
    return output, output_states   

def build_BRNN_v2(args, inputs, seq_len):
    
    with tf.variable_scope('BRNN_' + args.model):


        fw_stack_cell = tf.contrib.rnn.MultiRNNCell(
                        [rnn_cell(args) for i in range(args.num_layer)])
        
        bw_stack_cell = tf.contrib.rnn.MultiRNNCell(
                        [rnn_cell(args) for i in range(args.num_layer)])

        _initial_state_fw = fw_stack_cell.zero_state(args.batch_size, tf.float32)
        _initial_state_bw = bw_stack_cell.zero_state(args.batch_size, tf.float32)
        
        # tensor = [batch_size, time_step, input_feature]
        outputs, output_states =\
                        tf.nn.bidirectional_dynamic_rnn(fw_stack_cell, bw_stack_cell,
                                                        inputs=inputs,
                                                        sequence_length=seq_len,
                                                        initial_state_fw = _initial_state_fw,
                                                        initial_state_bw = _initial_state_bw)
        # rnn outputs 
        output_fw = tf.transpose(outputs[0], [1,0,2])
        output_bw = tf.transpose(outputs[1], [1,0,2])
        outputs = tf.concat([output_fw[-1], output_bw[-1]], axis=1)
       
        # rnn last states
        fw_states = output_states[0]
        bw_states = output_states[1]
        if args.model=='A':
            output_states = tf.concat([fw_states[-1][0], bw_states[-1][0]],axis=1) 
        else:
            output_states = tf.concat([fw_states[-1], bw_states[-1]],axis=1) 
    return outputs, output_states   

def build_RNN(args, input, seq_len):
    
    output = input
    for i in range(args.num_layer):
        scope = 'RNN_' + str(i+1)
        cell = rnn_cell(args)

        _initial_state = cell.zero_state(args.batch_size, tf.float32)

        # tensor = [batch_size, time_step, input_feature]
        output, output_states =\
                        tf.nn.dynamic_rnn(cell,
                                        inputs=output,
                                        sequence_length=seq_len,
                                        initial_state = _initial_state,
                                        scope=scope)
        

        if i == args.num_layer-1:
            output = tf.transpose(output, [1,0,2])
            output = output[-1]
    
    return output        

def fc_bn(args, inputs, dim_targets):

    if args.mode == 'train':
        is_training = True
    elif args.mode == 'test':
        is_training = False
    else:
        raise TypeError('model should be train or test')

    bn = tf.contrib.layers.batch_norm(inputs, is_training=is_training)
    fc = tf.contrib.layers.fully_connected(bn, dim_targets,
                    activation_fn=None)
    return fc
    
# LSTM model
class TSL_model():

    def __init__(self, args, num_steps, dim_inputs, dim_targets):
        self.args = args
        self.num_steps = num_steps
        self.dim_inputs = dim_inputs
        self.dim_targets = dim_targets
        self.build_graph(args, num_steps, dim_inputs, dim_targets)

    def build_graph(self, args, num_steps, dim_inputs, dim_targets):
        self.graph = tf.Graph()
        with self.graph.as_default():
            
            # input = [batch, num_steps, dim_inputs]
            self.inputs = tf.placeholder(tf.float32,
                            shape=(None, num_steps, dim_inputs))
           
            # target = [batch, dim_targets]
            self.targets = tf.placeholder(tf.float32,
                                shape=(None, dim_targets))
            self.seq_len = tf.placeholder(tf.int32, shape=(None))
 
            self.config = {'rnn_type': args.rnn_type,
                            'num_layer': args.num_layer,
                            'hidden_size': args.hidden_size,
                            'learning_rate': args.learning_rate,
                            'batch_size': args.batch_size}

            if self.args.rnn_type == 'bi':
                self.outputs, self.output_states = build_BRNN_v2(self.args, self.inputs, self.seq_len)
            elif self.args.rnn_type == 'uni':
                self.outputs = build_RNN(self.args, self.inputs, self.seq_len)

            print('states shape: {}'.format(self.output_states))
            self.predict = fc_bn(self.args, self.outputs, dim_targets)
          
            # decay learning rate
            self.loss = tf.reduce_mean(tf.square(self.predict- self.targets))
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = args.learning_rate
            self.learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                        args.epoch, 0.98, staircase=True) 
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                    global_step=global_step)

            self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.predict - self.targets)))
          
            # for counting parameters of model
            self.var_op = tf.global_variables()
            self.var_trainable_op = tf.trainable_variables()
            
            # initialization
            self.initial_op = tf.global_variables_initializer()
            
            # save variables of model
            self.saver = tf.train.Saver(tf.global_variables())
             
