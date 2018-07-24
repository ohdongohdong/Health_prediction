import os
import numpy as np
import tensorflow as tf
import time
import csv

def check_path_exists(path):
    '''
    check a path exists or not
    '''
    if isinstance(path, list):
        for p in path:
            if not os.path.exists(p):
                os.makedirs(p)
    else:
        if not os.path.exists(p):
            os.makedirs(path)

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def count_params(model, mode='trainable'):
    ''' count all parameters of a tensorflow graph
    '''
    if mode == 'all':
        num = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in model.var_op])
    elif mode == 'trainable':
        num = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in model.var_trainable_op])
    else:
        raise TypeError('mode should be all or trainable.')
    print('number of '+mode+' parameters: '+str(num))
    return num


def logging(model, logfile, each_epoch=0, epoch=100, batch=0, batch_epoch=256, 
                    loss=0.0, rmse=0.0, delta_time=0, mode='train'):
    ''' log the cost and error rate and time while training or testing
    '''  
    if mode == 'config':
        # write the info of configs
        with open(logfile, "a") as myfile:
            myfile.write('\n'+str(time.strftime('%X %x %Z'))+'\n')
            myfile.write('\n'+str(model.config)+'\n')

    elif mode == 'epoch':
        with open(logfile, "a") as myfile:
            myfile.write('\n[Epoch :{}]\n'.format(each_epoch+1))
    
    elif mode == 'batch':
        with open(logfile, "a") as myfile:
            myfile.write('batch: {}/{}, loss={:.4f}, rmse={:.4f}\n'.format(batch+1, batch_epoch, loss,rmse))

    elif mode =='train': 
        with open(logfile, "a") as myfile:
            myfile.write('==> Epoch: {}/{}, loss={:.4f}, rmse={:.4f}, epoch time : {}\n'\
                                .format(each_epoch+1, epoch, loss, rmse, delta_time)) 
    else:
        raise TypeError('mode should be write right.')
 
def next_batch(batch_size, data_set):
    '''
    split all of data to batch set
    '''  
    idx = np.arange(0, len(data_set[0]))
    np.random.shuffle(idx)
    idx = idx[:batch_size]

    batch_data_set = []
    for data in data_set:
        batch_data_set.append(np.asarray([data[i] for i in idx]))

    return batch_data_set

def get_tf_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth =True
    return config


def RMSE(pred, target, mode='RMSE'):
    ''' Root mean square error
    '''
    rmse_list = []
    for i in range(target.shape[1]):
        if mode == 'RMSE':
            rmse = np.sqrt(np.mean((target[:,i] - pred[:, i])**2))
        elif mode == 'NRMSE': 
            rmse = np.sqrt(np.sum((target[:, i] - pred[:, i]) ** 2) / np.sum(target[:, i]**2))
        else:
            raise TypeError('mode should be write right.')
        
        rmse_list.append(rmse)
    return rmse_list

def mean_rmse(rmse):
    return sum(rmse)/len(rmse)

def SaveRMSE(tag_targets, predic, test_targets):
    test_rmse = RMSE(predic, test_targets)
    # write rmse
    f = open('test_rmse.csv', 'w')
    wr = csv.writer(f)
    wr.writerow(tag_targets)
    wr.writerow(test_rmse)
    f.close()
    print('Saved RMSE of test data')
    return

def SaveNRMSE(tag_targets, predic, test_targets):
    test_rmse = NRMSE(predic, test_targets)
    # write rmse
    f = open('Normal_RMSE.csv', 'w')
    wr = csv.writer(f)
    wr.writerow(tag_targets)
    wr.writerow(test_rmse)
    f.close()
    print('Saved NRMSE of test data')
    return

def SaveReult(tag_targets, predic, test_targets):
    # write result
    test_targets = test_targets.tolist()
    predic = predic.tolist()

    f = open('result.csv', 'w')
    wr = csv.writer(f)
    wr.writerow([''] + tag_targets)
    for i in range(len(test_targets)):
        wr.writerow(['original'] + test_targets[i])
        wr.writerow(['prediction'] + predic[i])
    f.close()
    print('Saved result of test data')
    return
