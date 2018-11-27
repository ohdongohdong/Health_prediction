###########################
# Multi Ensemble Learning 
# this code is train and test mel model.
# load tsl model.
# train : 
# train model and save variables of model.
# test : 
# load model and evaluate.
#
# data : hospital admit patients
# input : rnn states by tsl model
# output : final predicted EMR data by ensemble
#
# by Donghoon Oh
###########################

import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

import numpy as np
import tensorflow as tf
import time
import datetime
from sklearn.externals import joblib

from data_preprocessing import read_data, padding, split_data, feature_scaler, feature_normalize, feature_logscale
from utils import *
from model.tsl.tsl import TSL_model
from model.ensemble.base import MEL_base_model
from model.ensemble.attented import MEL_attented_model
from model.ensemble.attention import MEL_attention_model
from model.ensemble.concat_base import MEL_concat_base
from model.ensemble.concat_attstate import MEL_concat_Attstate
from model.ensemble.concat_attention import MEL_concat_attention
from model.ensemble.hierarchy import MEL_hierarchical_model

# flags
from tensorflow.python.platform import flags
flags.DEFINE_string('mode', 'train', 'select train or test')
flags.DEFINE_string('model', 'baseline', 'select  model attention, baseline, concat_base, concat_attstate, concat_attention')
flags.DEFINE_string('data_type', 'state', 'select ensembel input: state or value')
flags.DEFINE_string('rnn_type', 'uni', 'select uni-directional or bi-directional')
flags.DEFINE_integer('num_layer', 3, 'set the layers of rnn')
flags.DEFINE_integer('batch_size', 256, 'set the batch size')
flags.DEFINE_integer('hidden_size', 512, 'set the hidden size of rnn cell')
flags.DEFINE_integer('epoch', 100, 'set the number of epochs')
flags.DEFINE_float('learning_rate', 0.001, 'set the learning rate')
flags.DEFINE_float('keep_prob', 1.0, 'set the dropout ')
flags.DEFINE_boolean('scaling', True, 'feature scaling')
flags.DEFINE_string('log_dir', '../log', 'set the log directory')

FLAGS = flags.FLAGS

# set arguments
mode = FLAGS.mode
model = FLAGS.model
data_type = FLAGS.data_type
rnn_type = FLAGS.rnn_type
num_layer = FLAGS.num_layer
batch_size = FLAGS.batch_size
hidden_size = FLAGS.hidden_size
learning_rate = FLAGS.learning_rate
scaling = FLAGS.scaling

if mode == 'test':
    epoch = 1
    keep_prob = 1.0
    is_training = False
else:
    epoch = FLAGS.epoch
    keep_prob = FLAGS.keep_prob
    is_training = True

# set path of log directory
log_dir = FLAGS.log_dir
ensemble_log_dir = os.path.join(log_dir, 'ensemble', 'logscale')
save_dir = os.path.join(ensemble_log_dir, model,'save')
result_dir = os.path.join(ensemble_log_dir, model, 'result')
logging_dir = os.path.join(ensemble_log_dir, model, 'logging')
check_path_exists([log_dir, ensemble_log_dir, save_dir, result_dir, logging_dir])

checkpoint_path = os.path.join(save_dir, 'ensemble.ckpt')

logfile = os.path.join(logging_dir, str(datetime.datetime.strftime(datetime.datetime.now(),
    '%Y-%m-%d_%H:%M:%S') + '_'+mode+'_ensemble_'+model+'.txt').replace(' ', '').replace('/', ''))

# Train
class Runner(object):
    # set configs
    def _default_configs(self):
        return {'mode' : mode,
                'model' : model,
                'data_type' : data_type,
                'rnn_type' : rnn_type,
                'num_layer' : num_layer,
                'batch_size': batch_size,
                'hidden_size': hidden_size,
                'epoch' : epoch, 
                'learning_rate': learning_rate,
                'keep_prob': keep_prob,
                'is_training': is_training
                }

    def load_data(self):
        # data preprocessing
        '''
        input : (batch, max_step, input_features)
        target : (batch, target_features)
        '''
        
        # read the data set
        input_set, target_set = read_data('fill')
    
        # padding 
        pad_input_set, seq_len = padding(input_set)
       
        tsl_model_type = 'ensemble'

        # split data set for model
        value_train, value_test, target_train, target_test, seq_train, seq_test = split_data(
                                                    pad_input_set, target_set, seq_len, tsl_model_type)
        
        if mode == 'train':
            return value_train, target_train, seq_train
        elif mode == 'test':
            return value_test, target_test, seq_test

    def load_tsl(self, args, model, input_set, target_set, seq_len_set): 
        '''load learned tsl model
            get predicted data for using ensemble input
        '''
        tsl_checkpoint_path = os.path.join(log_dir, 'tsl', args.rnn_type, args.model, 'save', 'tsl.ckpt')
        
        with tf.Session(graph=model.graph, config=get_tf_config()) as sess: 

            # initialization 
            sess.run(model.initial_op)
            epoch = 1
            # load tsl model 
            model.saver.restore(sess, tsl_checkpoint_path)
         
            print('\n[load tsl model {}]'.format(args.model))
            print('predicting')
            for each_epoch in range(epoch):
                # mini batch 
                batch_epoch = int(input_set.shape[0]/batch_size) 
                nrmse_list = []
                prediction = []
                state_list = [] # rnn state
                target_list = []
                for b in range(batch_epoch):
                   
                    batch_inputs = input_set[b*batch_size : (b+1)*batch_size]
                    batch_targets = target_set[b*batch_size : (b+1)*batch_size]
                    batch_seq_len = seq_len_set[b*batch_size : (b+1)*batch_size]
                    feed = {model.inputs:batch_inputs,
                            model.targets:batch_targets,
                            model.seq_len:batch_seq_len}

                    p, t, s = sess.run([model.predict,
                                        model.targets,
                                        model.output_states],
                                        feed_dict=feed)
                    batch_nrmse = RMSE(p, batch_targets, 'NRMSE')
                    prediction.extend(p)
                    state_list.extend(s)
                    target_list.extend(batch_targets)
                 
                    nr = mean_rmse(batch_nrmse)
                    nrmse_list.append(batch_nrmse)
                    
                    if b%10 == 0:
                        print('batch: {}/{}, nrmse={:.4f}'.format(b+1, batch_epoch, nr))
                        print(p[0]) 
                prediction = np.asarray(prediction)
                state_list = np.asarray(state_list)
                target_list = np.asarray(target_list)
                nrmse = np.asarray(nrmse_list).mean(axis=0)
        
                print('normalize rmse : ')
                print(nrmse)
                print('prediction shape : {}'.format(prediction.shape))
                print('state shape : {}'.format(state_list.shape))
                print('target shape : {}'.format(target_list.shape))
   
        print('finish loading tsl model {}\n'.format(args.model))
        return prediction, target_list, state_list 

    # fine-tuning train
    def fine_tuning(self, args, model,
                value_train_A, value_train_B, value_train_C,
                state_train_A, state_train_B, state_train_C,
                target_train):
        
        with tf.Session(graph=model.graph, config=get_tf_config()) as sess:
            # initialization 
            sess.run(model.initial_op)
            
            # load check point
            print('ensemble check point : {}'.format(checkpoint_path)) 
            model.saver.restore(sess, checkpoint_path)
            
            for each_epoch in range(epoch):
                start = time.time()
                print('\n[Epoch :{}]\n'.format(each_epoch+1))
                logging(model=model, logfile=logfile, each_epoch=each_epoch, mode='epoch')
                
                # mini batch 
                batch_epoch = int(target_train.shape[0]/batch_size)
                
                batch_loss = np.zeros(batch_epoch)
                rmse_list = []
                for b in range(batch_epoch):
                    
                    batch_value_A, batch_value_B, batch_value_C,\
                    batch_state_A, batch_state_B, batch_state_C,\
                    batch_targets = next_batch(batch_size,
                                           [value_train_A, value_train_B, value_train_C,
                                            state_train_A, state_train_B, state_train_C,
                                            target_train])

                    feed = {model.value_A:batch_value_A,
                            model.value_B:batch_value_B,
                            model.value_C:batch_value_C,
                            model.state_A:batch_state_A,
                            model.state_B:batch_state_B,
                            model.state_C:batch_state_C,
                            model.targets:batch_targets}

                    
                    _, l, r, p = sess.run([model.optimizer,
                                        model.loss, model.rmse,
                                        model.predict],
                                        feed_dict=feed)

                    batch_loss[b] = l
                    batch_nrmse = RMSE(p, batch_targets, 'NRMSE')
                    r = mean_rmse(batch_nrmse)
                    rmse_list.append(batch_nrmse)
                    if b%50 == 0:
                        print('batch: {}/{}, loss={:.3f}, rmse={:.3f}'.format(b+1, batch_epoch, l,r))
                        logging(model, logfile, batch=b, batch_epoch=batch_epoch, loss=l, rmse=r, mode='batch')
                
                loss = np.sum(batch_loss)/batch_epoch
                rmse = np.asarray(rmse_list).mean(axis=0)
                rmse_mean = mean_rmse(rmse) 

                delta_time = time.time()-start
                print('\n==> Epoch: {}/{}, loss={:.4f}, rmse={:.4f}, epoch time : {}\n'\
                                .format(each_epoch+1, epoch, loss, rmse_mean, delta_time))
                logging(model, logfile, each_epoch, epoch,
                            loss=loss, rmse=rmse_mean, delta_time=delta_time, mode='train')
                
                # save model by epoch
                model.saver.save(sess, checkpoint_path) 
                print('Prediction : ') 
                print(p[0])
                print('target : ') 
                print(batch_targets[0])
    
    # train
    def train(self, args, model,
                value_train_A, value_train_B, value_train_C,
                state_train_A, state_train_B, state_train_C,
                target_train):
        
        with tf.Session(graph=model.graph, config=get_tf_config()) as sess:
            # initialization 
            sess.run(model.initial_op)
            
            for each_epoch in range(epoch):
                start = time.time()
                print('\n[Epoch :{}]\n'.format(each_epoch+1))
                logging(model=model, logfile=logfile, each_epoch=each_epoch, mode='epoch')
                
                # mini batch 
                batch_epoch = int(target_train.shape[0]/batch_size)
                
                batch_loss = np.zeros(batch_epoch)
                rmse_list = []
                for b in range(batch_epoch):
                    
                    batch_value_A, batch_value_B, batch_value_C,\
                    batch_state_A, batch_state_B, batch_state_C,\
                    batch_targets = next_batch(batch_size,
                                           [value_train_A, value_train_B, value_train_C,
                                            state_train_A, state_train_B, state_train_C,
                                            target_train])

                    feed = {model.value_A:batch_value_A,
                            model.value_B:batch_value_B,
                            model.value_C:batch_value_C,
                            model.state_A:batch_state_A,
                            model.state_B:batch_state_B,
                            model.state_C:batch_state_C,
                            model.targets:batch_targets}

                    
                    _, l, r, p= sess.run([model.optimizer,
                                        model.loss, model.rmse,
                                        model.predict],
                                        feed_dict=feed)
                    batch_loss[b] = l
                    batch_nrmse = RMSE(p, batch_targets, 'NRMSE')
                    r = mean_rmse(batch_nrmse)
                    rmse_list.append(batch_nrmse)
                    if b%50 == 0:
                        print('batch: {}/{}, loss={:.3f}, rmse={:.3f}'.format(b+1, batch_epoch, l,r))
                        logging(model, logfile, batch=b, batch_epoch=batch_epoch, loss=l, rmse=r, mode='batch')
                
                loss = np.sum(batch_loss)/batch_epoch
                rmse = np.asarray(rmse_list).mean(axis=0)
                rmse_mean = mean_rmse(rmse) 

                delta_time = time.time()-start
                print('\n==> Epoch: {}/{}, loss={:.4f}, rmse={:.4f}, epoch time : {}\n'\
                                .format(each_epoch+1, epoch, loss, rmse_mean, delta_time))
                logging(model, logfile, each_epoch, epoch,
                            loss=loss, rmse=rmse_mean, delta_time=delta_time, mode='train')
                
                # save model by epoch
                model.saver.save(sess, checkpoint_path) 
                print('Prediction : ') 
                print(p[0])
                print('target : ') 
                print(batch_targets[0])

    def test(self, args, model,
            value_test_A, value_test_B, value_test_C,
            state_test_A, state_test_B, state_test_C,
            target_test):

        with tf.Session(graph=model.graph, config=get_tf_config()) as sess: 
            # initialization 
            sess.run(model.initial_op)
            
            # load check point
            print('ensemble check point : {}'.format(checkpoint_path)) 
            model.saver.restore(sess, checkpoint_path)
                     
            for each_epoch in range(epoch):
                start = time.time()
                print('\n[Epoch :{}]\n'.format(each_epoch+1))
                logging(model=model, logfile=logfile, each_epoch=each_epoch, mode='epoch')
                
                # mini batch 
                batch_epoch = int(target_test.shape[0]/batch_size)
                
                batch_loss = np.zeros(batch_epoch)
                rmse_list = []
                nrmse_list = []
                for b in range(batch_epoch):
                    ''' 
                    batch_value_A, batch_value_B, batch_value_C,\
                    batch_state_A, batch_state_B, batch_state_C,\
                    batch_targets = next_batch(batch_size,
                                           [value_test_A, value_test_B, value_test_C,
                                            state_test_A, state_test_B, state_test_C,
                                            target_test])
                    '''

                    batch_value_A = value_test_A[b*batch_size : (b+1)*batch_size]
                    batch_value_B = value_test_B[b*batch_size : (b+1)*batch_size]
                    batch_value_C = value_test_C[b*batch_size : (b+1)*batch_size]
                    
                    batch_state_A = state_test_A[b*batch_size : (b+1)*batch_size]
                    batch_state_B = state_test_B[b*batch_size : (b+1)*batch_size]
                    batch_state_C = state_test_C[b*batch_size : (b+1)*batch_size]
                    
                    batch_targets = target_test[b*batch_size : (b+1)*batch_size]
                    feed = {model.value_A:batch_value_A,
                            model.value_B:batch_value_B,
                            model.value_C:batch_value_C,
                            model.state_A:batch_state_A,
                            model.state_B:batch_state_B,
                            model.state_C:batch_state_C,
                            model.targets:batch_targets}

                    l, p, t = sess.run([model.loss,
                                        model.predict,
                                        model.targets],
                                        feed_dict=feed)
                    batch_loss[b] = l
                    batch_rmse = RMSE(p, batch_targets, 'RMSE')
                    batch_nrmse = RMSE(p, batch_targets, 'NRMSE')
                    r = mean_rmse(batch_rmse)
                    nr = mean_rmse(batch_nrmse)
                    rmse_list.append(batch_rmse)
                    nrmse_list.append(batch_nrmse)

                    if b%10 == 0:
                        print('batch: {}/{}, loss={:.4f}, rmse={:.4f}'.format(b+1, batch_epoch, l,r))
                        logging(model, logfile, batch=b, batch_epoch=batch_epoch, loss=l, rmse=r, mode='batch')
                
                loss = np.sum(batch_loss)/batch_epoch 
                rmse = np.asarray(rmse_list).mean(axis=0)
                rmse_mean = mean_rmse(rmse) 
                nrmse = np.asarray(nrmse_list).mean(axis=0)
                delta_time = time.time()-start
                print('\n==> Epoch: {}/{}, loss={:.4f}, rmse={:.4f}, epoch time : {}\n'\
                                .format(each_epoch+1, epoch, loss, rmse_mean, delta_time))
                logging(model, logfile, each_epoch, epoch,
                            loss=loss, rmse=rmse_mean, delta_time=delta_time, mode='train')
                
                with open(logfile, 'a') as myfile:
                    myfile.write('\nRMSE : \n')
                    for i,e in enumerate(rmse):
                        if not i == 0: 
                            myfile.write(', '.format(e))
                        myfile.write('{:.4f}'.format(e))
                
                    myfile.write('\nNormalize RMSE : \n')
                    for i,e in enumerate(nrmse):
                        if not i == 0: 
                            myfile.write(', '.format(e))
                        myfile.write('{:.4f}'.format(e))
                
                print('rmse : ')
                print(rmse)
                print('normalize rmse : ')
                print(nrmse)
                
                print('Prediction : ') 
                print(p[0:3])
                print('target : ') 
                print(batch_targets[0:3])
    # main
    def run(self):
        # set args
        args_dict = self._default_configs()
        args = dotdict(args_dict)
       
        # step 1
        # load data 
        input_set, target_set, seq_len_set = self.load_data()
        print('[model data set]') 
        print('shape of input : {}'.format(input_set.shape))
        print('shape of target : {}'.format(target_set.shape))
       
        # data parameters
        num_steps = input_set.shape[1]
        dim_inputs = input_set.shape[2]
        dim_targets = target_set.shape[1]

        # step 2
        # load TSL model
        # predict data by tsl
        
        args.model = 'A'
        tsl_A_model = TSL_model(args, num_steps, dim_inputs, dim_targets)
        value_A, target_set, state_A= self.load_tsl(args, tsl_A_model, input_set, target_set, seq_len_set)
        
        args.model = 'B'
        tsl_B_model = TSL_model(args, num_steps, dim_inputs, dim_targets)
        value_B, _, state_B = self.load_tsl(args, tsl_B_model, input_set, target_set, seq_len_set)
        
        args.model = 'C'
        tsl_C_model = TSL_model(args, num_steps, dim_inputs, dim_targets)
        value_C, _, state_C = self.load_tsl(args, tsl_C_model, input_set, target_set, seq_len_set)  
       
        # data parameters
        dim_state = state_A.shape[1]
        dim_feature = target_set.shape[1]

        # log troponin
        '''
        value_A = feature_logscale(value_A, sequence = False)
        value_B = feature_logscale(value_B, sequence = False)
        value_C = feature_logscale(value_C, sequence = False)
        '''
        # ensemble data scaling 
        if scaling == True:
            
            state_A = feature_scaler(args, state_A, 
                                    os.path.join(save_dir, 'scaler_state_A.save')) 
            state_B = feature_scaler(args, state_B, 
                                    os.path.join(save_dir, 'scaler_state_B.save'))
            state_C = feature_scaler(args, state_C, 
                                    os.path.join(save_dir, 'scaler_state_C.save'))
            value_A = feature_scaler(args,value_A, 
                                    os.path.join(save_dir, 'scaler_value_A.save')) 
            value_B = feature_scaler(args,value_B, 
                                    os.path.join(save_dir, 'scaler_value_B.save'))
            value_C = feature_scaler(args,value_C, 
                                    os.path.join(save_dir, 'scaler_value_C.save'))  
        
        # step 3
        # load MEL model
        args.model = model
        args.num_layer = 1
        
        if model == 'baseline':
            mel_model = MEL_base_model(args, dim_feature, dim_state, data_type)
        elif model == 'attention':
            mel_model = MEL_attention_model(args, dim_feature, dim_state, data_type)
        elif model == 'attented':
            mel_model = MEL_attented_model(args, dim_feature, dim_state, data_type)
        elif model == 'concat_base':
            mel_model = MEL_concat_base(args, dim_feature, dim_state)
        elif model == 'concat_attstate':
            mel_model = MEL_concat_Attstate(args, dim_feature, dim_state)
        elif model == 'concat_attention':
            mel_model = MEL_concat_attention(args, dim_feature, dim_state)
        elif model == 'autoencoder':
            mel_model = AutoEncoder(args, dim_feature, dim_state, data_type)
        elif model == 'hierarchy':
            mel_model = MEL_hierarchical_model(args, dim_feature, dim_state, data_type)

        # count the num of parameters
        num_params = count_params(mel_model, mode='trainable')
        all_num_params = count_params(mel_model, mode='all')
        mel_model.config['trainable params'] = num_params
        mel_model.config['all params'] = all_num_params
        print('\n[model information]\n')
        print(mel_model.config)  
        
        # setp 4
        # learning 
        logging(model=mel_model, logfile=logfile, mode='config')
        
        if mode == 'train':
            self.train(args, mel_model, 
                        value_A, value_B, value_C, 
                        state_A, state_B, state_C,
                        target_set)
        elif mode == 'test':
            self.test(args, mel_model, 
                        value_A, value_B, value_C, 
                        state_A, state_B, state_C,
                        target_set)

if __name__ == '__main__':
    runner = Runner()
    runner.run()
