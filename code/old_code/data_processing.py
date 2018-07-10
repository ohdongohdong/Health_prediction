import numpy as np
import csv

data_path = '../../data/old_data/data.npy'
tag_path = '../../data/old_data/tag.npy'

def LoadData():
    # Using data that had patient's feature value of all years(2002 ~ 2013)
    # load data
    data = np.load(data_path)
    tag = np.load(tag_path)

    return data, tag

def SplitData(data, tag, split):
    # copy from hgkim (I don't know why use only this features....)
    index_of_inputs = list(range(3, 10)) + list(range(12, 15)) + list(range(38, 40))
    index_of_targets = list(range(5, 10)) + list(range(12, 15))

    tag_inputs = tag[index_of_inputs].tolist()
    tag_targets = tag[index_of_targets].tolist()

    train_inputs = data[:split, :-1, index_of_inputs]
    train_targets = data[:split, -1, index_of_targets]

    test_inputs = data[split:, :-1, index_of_inputs]
    test_targets = data[split:, -1, index_of_targets]

    return train_inputs, train_targets, test_inputs, test_targets, tag_inputs, tag_targets

# each features RMSE
# len pred = # of patients
# len pred[0] = # of features
def RMSE(pred, target):
    rmse_list = []
    for i in range(target.shape[1]):
        rmse = np.sqrt(np.mean((target[:,i] - pred[:, i])**2))
        rmse_list.append(rmse)
    return rmse_list

def NRMSE(pred, target):
    rmse_list = []
    for i in range(target.shape[1]):
        rmse = np.sqrt(np.sum((target[:, i] - pred[:, i]) ** 2) / np.sum(target[:, i]**2))
        rmse_list.append(rmse)
    return rmse_list

def mean_rmse(rmse):
    n = 0
    for i in range(len(rmse)):
        n += rmse[i]
    return n/len(rmse)

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
