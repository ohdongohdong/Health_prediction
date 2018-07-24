###########################
# Multiple Ensemble Learning 
# this code is train mel model.
# train model and save variables of model.
#
# data : hospital admit patients
# input : predicted feature vector of tsl
# output : final predict feature output
#
# by Donghoon Oh
###########################

import os
import numpy as np
import tensorflow as tf
import time
import datetime

from data_preprocessing import read_data, padding, split_data
from utils import *
from model.mel_hierarchy import Model

# flags
from tensorflow.python.platform import flags
flags.DEFINE_string('mode', 'train', 'select train or test')
flags.DEFINE_string('model', 'hierarchy', 'select hospital model hierarchy, retain')
flags.DEFINE_integer('num_layer', 3, 'set the layers of rnn')
flags.DEFINE_integer('batch_size', 256, 'set the batch size')
flags.DEFINE_integer('hidden_size', 512, 'set the hidden size of rnn cell')
flags.DEFINE_integer('epoch', 100, 'set the number of epochs')
flags.DEFINE_float('learning_rate', 0.001, 'set the learning rate')
flags.DEFINE_string('log_dir', '../log/ensemble', 'set the log directory')
flags.DEFINE_string('data_dir', '../data/ensemble', 'set the log directory')

FLAGS = flags.FLAGS

# set arguments
mode = FLAGS.mode
model = FLAGS.model
num_layer = FLAGS.num_layer
batch_size = FLAGS.batch_size
hidden_size = FLAGS.hidden_size
epoch = FLAGS.epoch
learning_rate = FLAGS.learning_rate

# set path of log directory
log_dir = FLAGS.log_dir
data_dir = FLAGS.data_dir
save_dir = os.path.join(log_dir, model,'save')
result_dir = os.path.join(log_dir, model, 'result')
logging_dir = os.path.join(log_dir, model, 'logging')
check_path_exists([log_dir, save_dir, result_dir, logging_dir])

checkpoint_path = os.path.join(save_dir, 'ensemble.ckpt')

logfile = os.path.join(logging_dir, str(datetime.datetime.strftime(datetime.datetime.now(),
    '%Y-%m-%d_%H:%M:%S') + '_ensemble_'+model+'.txt').replace(' ', '').replace('/', ''))

data_path = os.path.join(data_dir, mode)

if mode=='train':
    is_training = True
elif mode=='test':
    is_training = False

# Train
class Runner(object):
    # set configs
    def _default_configs(self):
        return {'mode' : mode,
                'model' : model,
                'num_layer' : num_layer,
                'batch_size': batch_size,
                'hidden_size': hidden_size,
                'learning_rate': learning_rate,
                }
    
    # train
    def run(self):
        # set args
        args_dict = self._default_configs()
        args = dotdict(args_dict)
        args.is_training = is_training

        # [step 1] 
        # read the data set
        # by tsl model
        input_train_A = np.load(os.path.join(data_path, 'ensemble_input_A.npy'))
        input_train_B = np.load(os.path.join(data_path, 'ensemble_input_B.npy'))
        input_train_C = np.load(os.path.join(data_path, 'ensemble_input_C.npy'))
        target_train = np.load(os.path.join(data_path, 'ensemble_target.npy'))

        print('\n[shape of data set]')
        print('input A : {}'.format(input_train_A.shape))
        print('input B : {}'.format(input_train_B.shape))
        print('input C : {}'.format(input_train_C.shape))
        print('target : {}'.format(target_train.shape))
        
        # data parameters
        dim_feature = target_train.shape[1]

        # [step 2]
        # ensemble model 
        model = Model(args, dim_feature)

        # count the num of parameters
        num_params = count_params(model, mode='trainable')
        all_num_params = count_params(model, mode='all')
        model.config['trainable params'] = num_params
        model.config['all params'] = all_num_params
        print('\n[model information]\n')
        print(model.config)  
        
        # step 3
        # learning
        with tf.Session(graph=model.graph, config=get_tf_config()) as sess:
            # initialization 
            sess.run(model.initial_op)
            
            logging(model=model, logfile=logfile, mode='config')

            for each_epoch in range(epoch):
                start = time.time()
                print('\n[Epoch :{}]\n'.format(each_epoch+1))
                logging(model=model, logfile=logfile, each_epoch=each_epoch, mode='epoch')
                
                # mini batch 
                batch_epoch = int(target_train.shape[0]/batch_size)
                
                batch_loss = np.zeros(batch_epoch)
                rmse_list = []
                for b in range(batch_epoch):
                    
                    # split batch data set
                    '''
                    batch_inputs_A = input_train_A[b*batch_size : (b+1)*batch_size]
                    batch_inputs_B = input_train_B[b*batch_size : (b+1)*batch_size]
                    batch_inputs_C = input_train_C[b*batch_size : (b+1)*batch_size]
                    
                    batch_targets = target_train[b*batch_size : (b+1)*batch_size]
                    '''
                    batch_inputs_A, batch_inputs_B, batch_inputs_C, batch_targets = next_batch_v2(
                                       batch_size,
                                       [input_train_A, input_train_B, input_train_C, target_train])

                    feed = {model.inputs_A:batch_inputs_A,
                            model.inputs_B:batch_inputs_B,
                            model.inputs_C:batch_inputs_C,
                            model.targets:batch_targets}

                    _, l, r, p = sess.run([model.optimizer,model.loss, model.rmse,
                                        model.predict],
                                        feed_dict=feed)
                    batch_loss[b] = l
                    batch_rmse = RMSE(p, batch_targets, 'RMSE')
                    r = mean_rmse(batch_rmse)
                    rmse_list.append(batch_rmse)
                    if b%50 == 0:
                        print('batch: {}/{}, loss={:.4f}, rmse={:.4f}'.format(b+1, batch_epoch, l,r))
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
                print('batch RMSE : ')
                print(rmse)
                print('\nPrediction : ') 
                print(p[0])
                print('target : ') 
                print(batch_targets[0])

if __name__ == '__main__':
    runner = Runner()
    runner.run()
