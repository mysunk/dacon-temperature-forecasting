# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 21:24:21 2020

@author: guseh
"""
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
try:
    import cPickle as pickle
except BaseException:
    import pickle
import os
from configparser import ConfigParser
from sklearn.metrics import mean_squared_error

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

def add_profile(data, feature_names):
    
    profiles = []
    for f in feature_names:
        profile = pd.DataFrame(data.loc[:,[f]])
        profile = np.reshape(profile.values, (-1,6*24)) # 하루치
        profiles.append(profile)
    
    profile = np.concatenate(profiles,axis=1)
    
    for i in range(1,(int)(data.shape[0]/144)): # 첫 날은 제외
        new_data = pd.concat([data.iloc[i*144:(i+1)*144].reset_index(drop=True), pd.DataFrame(np.tile(profile[i-1,:], (144, 1)))],axis=1)
        if i==1:
            train_1_p = new_data
        else:
            train_1_p = pd.concat([train_1_p,new_data],axis=0)
    return train_1_p

def add_profile_v2(data, features, N):
    if N!=0:
        new = data.iloc[N:,:].reset_index(drop=True) # 앞에 N개 자름
        additional = pd.DataFrame(np.zeros((data.shape[0]-N,len(features)*N)))
        for i in range(N,data.shape[0]):
            additional.iloc[i-N,:]= np.ravel(data.loc[i-N:i-1,features].values) # 한줄로 만들어서 옆에 붙임
        new = pd.concat([new, additional],axis=1)
    else:
        new = data
    return new

def add_profile_v3(data, features, N):
    new = data.iloc[N:,:].reset_index() # 앞에 N개 자름
    additional = pd.DataFrame(np.zeros((data.shape[0]-N,len(features)*N)))
    for i in range(N,data.shape[0]):
        additional.iloc[i-1]= data.loc[i-1,features].values # 한줄로 만들어서 옆에 붙임
    new = pd.concat([new, additional],axis=1)
    return new

def add_profile_v4(data, feature, N):
    if N!=0:
        for i in range(N):
            additional = np.zeros((data.shape[0]))
            additional[i+1:] = data.loc[:data.shape[0]-i-2,feature].values
            additional = pd.DataFrame(additional,columns=[feature+'_'+str(i+1)])
            data = pd.concat([data, additional],axis=1)
    return data


def mse_AIFrenz_lgb(y_pred, train_data): # custom score function
    '''
    y_true: 실제 값
    y_pred: 예측 값
    '''
    y_true = train_data.get_label()
    diff = abs(y_true - y_pred)

    less_then_one = np.where(diff < 1, 0, diff)
    
    # multi-column일 경우에도 계산 할 수 있도록 np.average를 한번 더 씌움
    score = np.average(np.average(less_then_one ** 2, axis = 0))
    return 'mse_modified', score, False

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

def mse(y_pred, train_data): # custom score function
    '''
    y_true: 실제 값
    y_pred: 예측 값
    '''
    y_true = train_data.get_label()
    
    # multi-column일 경우에도 계산 할 수 있도록 np.average를 한번 더 씌움
    score = mean_squared_error(y_true, y_pred)
    return 'mse_modified', score, False

def mse_original(y_pred, train_data): # custom score function
    '''
    y_true: 실제 값
    y_pred: 예측 값
    '''
    y_true = train_data.get_label()
    
    # multi-column일 경우에도 계산 할 수 있도록 np.average를 한번 더 씌움
    score = mean_squared_error(y_true, y_pred)
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
        with open('results/'+ name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    except FileNotFoundError:
        os.mkdir('results')
        with open('results/'+ name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)        

def load_obj(name):
    with open('results/' + name + '.pkl', 'rb') as f:
        trials = pickle.load(f)
        trials = sorted(trials, key=lambda k: k['loss'])
        return trials
    
def lgb_eval_function(pred, train):
    diff = abs(pred - train)
    less_then_one = np.where(diff < 1, 0, diff)
    # multi-column일 경우에도 계산 할 수 있도록 np.average를 한번 더 씌움
    score = np.average(np.average(less_then_one ** 2, axis = 0))
    return 'custom_mse', score, False

def make_param_int(param, key_names):
    for key, value in param.items():
        if key in key_names:
            param[key] = int(param[key])
    return param

def irradiance_difference(data):
    data_diff = np.zeros(data.shape)
    
    for i in range(1,len(data)-1):
        if i%144==143 or i%144 == 0:
            # print(i)
            data_diff[i] = 0
        else:
            data_diff[i] = data[i+1] - data[i]
        
    return data_diff

def irradiance_difference_v2(data):
    data_diff = np.zeros(data.shape)
    
    for i in range(1,len(data)-1):
        if i%144==143 or i%144 == 0:
            # print(i)
            data_diff[i] = 0
        else:
            
            data_diff[i] = data[i+1] - data[i]
        
    return data_diff

def process_dataset(data):
    data['time'] = data.id.values % 144
    data['X11_diff'] = irradiance_difference(data.X11.values) # 누적을 difference로 바꿈
    data['X34_diff'] = irradiance_difference(data.X34.values)
    
    N_T = 12
    N_S = 20
    data = data.loc[:,['time','X00','X07','X30','X31','X34','X34_diff']]
    data = add_profile_v4(data, 'X31',N_T) # 온도
    data = add_profile_v4(data, 'X34_diff',N_S) # 일사량
    return data

def load_dataset_v2(data_type, N_T, N_S, do_norm):
    train = pd.read_csv('data_raw/train.csv')
    train_label = train.loc[:,'Y00':'Y18']
    train = train.loc[:,'id':'X39']
    time = train.id.values % 144
    test = pd.read_csv('data_raw/test.csv')
    time_test = test.id.values % 144
    tmp = pd.read_csv('data_raw/train_X34_diff.csv')
    tmp2 = pd.read_csv('data_raw/test_X34_diff.csv')
    train['X34_diff'] = tmp.iloc[:,1].values
    test['X34_diff'] = tmp2.iloc[:,1].values
    train['X39_diff'] = irradiance_difference(train.X39.values)
    test['X39_diff'] = irradiance_difference(test.X39.values)
    
    if do_norm:
        scaler = StandardScaler()
        train.loc[:,:] = scaler.fit_transform(train.values)
        test.loc[:,:] = scaler.transform(test.values)
    
    train['time'] = time
    test['time'] = time_test
    
    train = pd.concat([train, test],axis=0).reset_index(drop = True)
    # train = train.loc[:,['time','X00','X07','X30','X31','X34','X34_diff','X39','X39_diff']]
    train = add_profile_v4(train, 'X31',N_T) # 온도
    train = add_profile_v4(train, 'X34_diff',N_S) # 일사량
    # train = add_profile_v4(train, 'X39_diff',N_R) # 강수량
    
    if data_type == 'train1':
        train = train.iloc[:4320,:]
        train_label = train_label.loc[:4320-1,'Y00':'Y17']
        return train, train_label
    elif data_type == 'train2':
        train = train.iloc[4320:4752,:]
        train_label = train_label.loc[4320:,'Y18']
        return train, train_label
    elif data_type == 'test':
        test = train.iloc[4752:,:]
        return test
    elif data_type == 'train':
        return train.iloc[:4752,:]

