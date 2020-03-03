# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 21:24:21 2020

@author: guseh
"""

import pandas as pd
import numpy as np
try:
    import cPickle as pickle
except BaseException:
    import pickle
import os

def load_dataset(path):
    train = pd.read_csv(path+'train.csv')
    test = pd.read_csv(path+'test.csv')
    sample = pd.read_csv(path+'sample_submission.csv')
    
    # split data and label
    train_1 = train.iloc[:24*6*30,:41] # 30일간 X00 ~ X39
    train_2 = train.iloc[24*6*30:,:41] # 3일간 X00 ~ X39
    train_label_1 = train.iloc[:24*6*30,41:-1] # 30일간 Y00 ~ Y17
    train_label_1.insert(0,'id',train.id[:24*6*30])
    train_label_2 = train.iloc[24*6*30:,-1] # 3일간 Y18
    train_label_2 = pd.concat( [train.id[24*6*30:],train_label_2], axis=1)
    
    return train_1, train_2, train_label_1, train_label_2, test, sample

def mse_AIFrenz(y_true, y_pred):
    '''
    y_true: 실제 값
    y_pred: 예측 값
    '''
    diff = abs(y_true - y_pred)
    
    less_then_one = np.where(diff < 1, 0, diff)
    
    # multi-column일 경우에도 계산 할 수 있도록 np.average를 한번 더 씌움
    score = np.average(np.average(less_then_one ** 2, axis = 0))
    
    return score

def make_day_sample(data):
    if 'id' in data.columns:
        data = data.drop(columns='id')
    r, c = data.shape
    days = int(r/144)
    data_conv = np.zeros((days, 144*c))
    for i in range(days):
        tmp = data.iloc[144*i:144*(i+1),:]
        data_conv[i,:] = np.ravel(tmp.values)
    return data_conv

def make_day_label(label):
    if 'id' in label.columns:
        label = label.drop(columns='id')
    label_conv = []
    num_output = 18
    for i in range(label.shape[1]):
        tmp = label.iloc[:,i].values
        label_conv.append(np.reshape(tmp,(-1,144)))
    return label_conv

def save_obj(obj, name):
    try:
        with open('trials/'+ name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    except FileNotFoundError:
        os.mkdir('trials')
        with open('trials/'+ name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)        

def load_obj(name):
    with open('trials/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def lgb_eval_function(pred, train):
    diff = abs(pred - train)
    less_then_one = np.where(diff < 1, 0, diff)
    # multi-column일 경우에도 계산 할 수 있도록 np.average를 한번 더 씌움
    score = np.average(np.average(less_then_one ** 2, axis = 0))
    return 'custom_mse', score, False