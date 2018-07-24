###########################
# Time Series Learning 
# this code is test tsl model.
# select tsl model.
# load trained model and test.
#
# data : hospital admit patients
# input : previous EMR data
# output : predicted next EMR data
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
from model.tsl import Model

# flags
from tensorflow.python.platform import flags
flags.DEFINE_string('mode', 'test', 'train or test')
flags.DEFINE_string('model', 'A', 'select hospital model A, B, C')
flags.DEFINE_integer('num_layer', 3, 'set the layers of rnn')
flags.DEFINE_integer('batch_size', 256, 'set the batch size')
flags.DEFINE_integer('hidden_size', 512, 'set the hidden size of rnn cell')
flags.DEFINE_integer('epoch', 1, 'set the number of epochs')
flags.DEFINE_float('learning_rate', 0.001, 'set the learning rate')
flags.DEFINE_string('log_dir', '../log/tsl', 'set the log directory')

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
save_dir = os.path.join(log_dir, model,'save')
result_dir = os.path.join(log_dir, model, 'result')
logging_dir = os.path.join(log_dir, model, 'logging')
check_path_exists([log_dir, save_dir, result_dir, logging_dir])

checkpoint_path = os.path.join(save_dir, 'tsl.ckpt')

logfile = os.path.join(logging_dir, str(datetime.datetime.strftime(datetime.datetime.now(),
    '%Y-%m-%d_%H:%M:%S') + '_TEST_tsl_'+model+'.txt').replace(' ', '').replace('/', ''))

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
    
    # test
    def run(self):
        # set args
        args_dict = self._default_configs()
        args = dotdict(args_dict)
        
        # [step 1]
        # data preprocessing
        '''
        input : (batch, max_step, input_features)
        target : (batch, target_features)
        '''
        
        # read the data set
        input_set, target_set = read_data('fill')
        # padding 
        pad_input_set, seq_len = padding(input_set)
        
        # split data set for model
        input_train, input_test, target_train, target_test, seq_train, seq_test = split_data(
                                                    pad_input_set, target_set, seq_len, args.model)
 
        # data parameters
        num_steps = input_test.shape[1]
        dim_inputs = input_test.shape[2]
        dim_targets = target_test.shape[1]

        # [step 2]
        # load TSL model
        model = Model(args, num_steps, dim_inputs, dim_targets)

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
           
            # load check point
            model.saver.restore(sess, checkpoint_path)

            logging(model=model, logfile=logfile, mode='config')

            
            for each_epoch in range(epoch):
                start = time.time()
                print('\n[Epoch :{}]\n'.format(each_epoch+1))
                logging(model=model, logfile=logfile, each_epoch=each_epoch, mode='epoch')
                
                # mini batch 
                batch_epoch = int(input_test.shape[0]/batch_size)
                
                batch_loss = np.zeros(batch_epoch)
                rmse_list = []
                for b in range(batch_epoch):
                    
                    batch_inputs, batch_targets, batch_seq_len = next_batch(
                                    batch_size, [input_test, target_test, seq_test])  
                    
                    feed = {model.inputs:batch_inputs,
                            model.targets:batch_targets,
                            model.seq_len:batch_seq_len}

                    l, r, p, t = sess.run([model.loss, model.rmse,
                                        model.predict,model.targets],
                                        feed_dict=feed)
                    batch_loss[b] = l
                    batch_rmse = RMSE(p, batch_targets, 'RMSE')
                    r = mean_rmse(batch_rmse)
                    rmse_list.append(batch_rmse)

                    if b%10 == 0:
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
                with open(logfile, 'a') as myfile:
                    myfile.write('\nRMSE : \n')
                    for i,e in enumerate(rmse):
                        if not i == 0: 
                            myfile.write(', '.format(e))
                        myfile.write('{:.4f}'.format(e))
                
                print('rmse : ')
                print(rmse)
                # save model by epoch
                print('Prediction : ') 
                print(p[0])
                print('target : ') 
                print(batch_targets[0])

if __name__ == '__main__':
    runner = Runner()
    runner.run()
