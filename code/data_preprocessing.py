"""
2018-06-28

health prediction
data processing

1. select all of data or dropna data or fill data
2. padding 
3. split train and test data set

by Donghoon Oh
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, scale, maxabs_scale, robust_scale,StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler

data_dir = '../data/'

all_data_path = os.path.join(data_dir, 'all_hadm_data')
dropna_data_path = os.path.join(data_dir, 'dropna_hadm_data')
fill_data_path = os.path.join(data_dir, 'fill_hadm_data')

# Convert hdam csv data to numpy array data set
def csv2npy(data_select):

#input feature : age, gender, 
#                glucose,
#                total_chol, hdl_c, chol_ratio, ldl_c,
#                triglyseride, troponin,
#                temp, heart_rate,
#                sysbp, diasbp (blood pressure)
#
#target feature : input feature without age, gender
#
#input : [time_step, dim_input_feature]
#target : [dim_target_feature]

    input_feat = list(range(0,12))
    target_feat = list(range(2,13))
    #print(select_feat)

# select data set 
    if data_select == 'dropna':
        data_path = dropna_data_path
    elif data_select == 'all':
        data_path = all_data_path
    elif data_select == 'fill':
        data_path = fill_data_path
    
    input_set = []
    target_set = []
    print('\nStart read data of hadm\n')
    for i,f in enumerate(os.listdir(data_path)):
        if i%1000 == 0:
            print('{}/{}'.format(i, len(os.listdir(data_path))))
        data = np.genfromtxt(os.path.join(data_path,f), delimiter=',', skip_header=1, filling_values=-1)[:,3:]
        if not len(data) > 201: # set max time step
            for x in range(len(data)):
                for y in range(len(data[0])): 
                    if data[x, y] <= 0.0:
                        if y == 9:
                            print(data[x])
                            data[x,y] = 36.946
                        else:
                            data[x,y] = data[x-1, y]
            input_set.append(data[:-1, :])
            target_set.append(data[-1, 2:])

    input_set = np.asarray(input_set)
    target_set = np.asarray(target_set)

    print('shape input : {}'.format(input_set.shape))
    print('shape target : {}'.format(target_set.shape))

    # make npy type data sets
    np.save(os.path.join(data_dir,(data_select+'_input')), input_set)
    np.save(os.path.join(data_dir,(data_select+'_target')), target_set)
 
    print('Finished converting csv to numpy array data set')

# Read numpy array type data set
# Make input and target data set
def read_data(data_select):
    print('Read data set')
    input_data_path = os.path.join(data_dir, (data_select+'_input.npy'))
    target_data_path = os.path.join(data_dir, (data_select+'_target.npy'))

    input_data_set = np.load(input_data_path)
    target_data_set = np.load(target_data_path)

    #input_data_set = feature_logscale(input_data_set)
    #input_data_set = feature_scaling(input_data_set)
    #target_data_set = feature_logscale(target_data_set, sequence=False)
    
    print('shape input : {}'.format(input_data_set.shape))
    print('shape target : {}'.format(target_data_set.shape))
   
    return input_data_set, target_data_set

def feature_logscale(data_set, sequence=True):
    with np.errstate(invalid='raise'):
    #with np.errstate(divide='raise'):
        if sequence == True: 
            log_data_set = [] 
             
            for data in data_set:
                log_data_set.append(np.hstack(\
                            (data[:,0:8], np.reshape(np.log(data[:,8]),(-1,1)),\
                                data[:,9:])))
            log_data_set = np.array(log_data_set) 
        else:
            for d in data_set:
                if d[6] <= 0:
                    print(d)
            log_data_set = np.hstack(\
                        (data_set[:, 0:6], np.reshape(np.log(data_set[:,6]),(-1,1)),\
                         data_set[:,7:]))

    return log_data_set

def feature_normalize(data_set):
    
    if len(data_set.shape) == 2:
        # for ensemble inputs
        # [num_data, dim_feature]
        #data_set = scale(data_set, axis=0)
        data_set = normalize(data_set)
        #data_set = robust_scale(data_set, axis=0)
    
    else:
        #for tsl inputs (time series) 
        # [num_data, time_step, dim_feature]
        for i in range(len(data_set)):
            data_set[i] = normalize(data_set[i], axis=0)

    normalized_data_set = data_set
    print('normalized data set : {}'.format(normalized_data_set.shape))
    
    return normalized_data_set

def feature_scaling(data_set):
    for i in range(len(data_set)):
        data_set[i] = scale(data_set[i], axis=0)

    return data_set

def feature_scaler(args, data_set, scaler_path):

    # for ensemble inputs
    # [num_data, dim_feature]

    if args.mode == 'train':
        scaler = StandardScaler().fit(data_set)
        #scaler = MaxAbsScaler().fit(data_set)
        #scaler = MinMaxScaler().fit(data_set) 
        #scaler = RobustScaler().fit(data_set) 
        joblib.dump(scaler, scaler_path)
    elif args.mode == 'test':
        scaler = joblib.load(scaler_path)
    
    scaled_data = scaler.transform(data_set)
    return scaled_data

# zero padding and get real sequence length
def padding(data_set):

    max_ts = 0
    for data in data_set:
        if max_ts < len(data):
            max_ts = len(data)
    print('max time step : {}'.format(max_ts))

    print('padding input data set')
    seq_len = []
    pad_data_set = []
    for i in range(len(data_set)):
        seq_len.append(len(data_set[i]))
        tmp = np.zeros((max_ts, data_set[i].shape[1]))
        tmp[:data_set[i].shape[0],:data_set[i].shape[1]] = data_set[i]
        pad_data_set.append(tmp)

    pad_data_set = np.asarray(pad_data_set)
    print('shape of total input data : {}'.format(pad_data_set.shape))
    return pad_data_set, seq_len

# split train and test set
# split train set to each of models train set
def split_data(input_set, target_set, seq_len, model):

    print('split train and test set')
    input_train, input_test, target_train, target_test, seq_train, seq_test = train_test_split(
            input_set, target_set, seq_len, test_size=0.1, random_state=42)
    
    print('shape of input train : {}'.format(input_train.shape))
    print('shape of target train : {}'.format(target_train.shape))
    print('shape of input test : {}'.format(input_test.shape))
    print('shape of target test : {}'.format(target_test.shape))

    if not model == 'ensemble':

        # num of models of hospital : 3
        split_flag = int(len(input_train)/3)
        print(split_flag)

        if model=='A':
            input_train = input_train[:split_flag]
            target_train = target_train[:split_flag]
            seq_train = seq_train[:split_flag]

        elif model=='B':
            input_train = input_train[split_flag:split_flag*2]
            target_train = target_train[split_flag:split_flag*2]
            seq_train = seq_train[split_flag:split_flag*2]

        elif model=='C':
            input_train = input_train[split_flag*2:]
            target_train = target_train[split_flag*2:]
            seq_train = seq_train[split_flag*2:]
     
        # make npy type data sets
        #np.save(os.path.join(data_dir,(data_select+'_input')), input_set)
        #np.save(os.path.join(data_dir,(data_select+'_target')), target_set)
        
    
    return input_train, input_test, target_train, target_test, seq_train, seq_test

#csv2npy('fill')
'''
csv2npy('fill')
input_set, target_set = read_data('fill')
pad_input_set, seq_len = padding(input_set)

input_train, input_test, target_train, target_test, seq_train, seq_test = split_data(pad_input_set, target_set, seq_len,'A')
'''
