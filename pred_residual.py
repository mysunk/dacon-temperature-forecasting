# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 15:55:26 2020

@author: guseh
"""
from util import*
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# clf
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor


def rf_param():
    rf_param =  {
        'max_depth':                5,
        'max_features':             9,
        'n_estimators':             643,
        'min_samples_leaf':         1,
        'min_samples_split':        6,
        'random_state' :            0,
       }
    return rf_param

label = 'Y16'
result = []
method = 'rf'
nfold = 10
random_states = [0]
for seeds in random_states:
    # user
    #trials = load_obj(label+'_residual_'+method)
    #trials = sorted(trials, key=lambda k: k['loss'])
    #param = trials[0]['params']
    param = rf_param()
    param['random_state']=seeds
    # param['max_depth'] = 2
    
    train, train_label = load_dataset_v2('train2',12, 20, True)
    del train['time']
    
    data = []
    data.append(np.load('data_pre/'+label+'_pred_3day_svr.npy'))
    data.append(np.load('data_pre/'+label+'_pred_3day_rf.npy'))
    data.append(np.load('data_pre/'+label+'_pred_3day_lgb.npy'))
    
    data = np.mean(data, axis=0)
    train_label = train_label.values - data
    train = train.values
    
    test = load_dataset_v2('test',12, 20, True)
    del test['time']
    test = test.values
    
    val = np.zeros((432))
    preds = []
    if nfold==0:
        if method == 'lgb':
            dtrain = lgb.Dataset(train, label=train_label)
            model = lgb.train(param, train_set = dtrain,valid_sets = [dtrain], num_boost_round=1000,verbose_eval=True)
        elif method == 'svr':
            model =MultiOutputRegressor(SVR(**param))
            model.fit(train, train_label)
        elif method == 'rf':
            model = RandomForestRegressor(n_jobs=-1,**param)
            model.fit(train, train_label)
        y_pred = model.predict(train)
        test_pred = model.predict(test)
        preds.append(test_pred)
        val = model.predict(train)
    else:
        losses = np.zeros((nfold,2)) # 0:train, 1:val
        kf = KFold(n_splits=nfold, random_state=None, shuffle=False)
        for i, (train_index, test_index) in enumerate(kf.split(train, train_label)):
            print(i,'th fold training')
            # train test split
            x_train = train[train_index]
            y_train = train_label[train_index]
            x_test = train[test_index]
            y_test = train_label[test_index]
            
            # w.r.t method
            if method == 'lgb':
                param['metric']='l2'
                dtrain = lgb.Dataset(x_train, label=y_train)
                dvalid = lgb.Dataset(x_test, label=y_test)
                model = lgb.train(param, train_set = dtrain,valid_sets = [dtrain, dvalid], num_boost_round=1000,verbose_eval=True,
                                         early_stopping_rounds=10)
            elif method == 'svr':
                model = SVR(**param)
                model.fit(x_train, y_train)
            elif method == 'rf':
                model = RandomForestRegressor(n_jobs=-1,**param)
                model.fit(x_train, y_train)
            # for evaluation
            train_pred = model.predict(x_train)
            valid_pred = model.predict(x_test)
            val[test_index] = valid_pred
            losses[i,0]= mean_squared_error(y_train, train_pred)
            losses[i,1]= mean_squared_error(y_test, valid_pred)
            # test
            test_pred = model.predict(test)
            preds.append(test_pred)
    result.append(np.mean(preds,axis=0))
    
#%% 3일치 결과
plt.plot(val)
plt.plot(train_label)

#%% 최종 예측
svr = np.load('data_pre/'+label+'_pred_80day_svr.npy')
lgb = np.load('data_pre/'+label+'_pred_80day_lgb.npy')
rf = np.load('data_pre/'+label+'_pred_80day_rf.npy')

data = [svr, rf, lgb]
data = np.mean(data, axis=0)
summed = data + np.mean(result,axis=0)
ref_prev = pd.read_csv('submit/submit_11.csv')
ref_prev_bad = pd.read_csv('submit/submit_18.csv')
ref = pd.read_csv('submit/sample_submission_v40.csv')
plt.plot(ref.Y18.values)
plt.plot(ref_prev_bad.Y18.values)
plt.plot(ref_prev.Y18.values)
plt.plot(summed)
plt.plot(y_pred)
plt.legend(['ref','ref_prev_bad','ref_prev','proposed','ensemble'])
y_pred = summed * 0.2 + ref.Y18.values * 0.8
mean_squared_error(ref.Y18.values, y_pred)
# np.save('pred_1.csv',summed)
#%% ensemble
preds = []
preds.append(np.load('pred_1.csv.npy'))
preds.append(np.load('pred_2.csv.npy'))
preds.append(ref.Y18.values)
y_pred = np.dot(np.array([0.5, 0.5]),preds)
mean_squared_error(ref.Y18.values, y_pred)

#%%
# plt.plot(preds[0])
# plt.plot(preds[1])
plt.plot(y_pred)
plt.plot(ref.Y18.values)
plt.plot(ref_prev.Y18.values)
plt.plot(ref_bad.Y18.values)
plt.legend(['pred','ref','ref_prev','ref_bad'])

#%%
ref = pd.read_csv('submit/sample_submission_v37.csv')
mean_squared_error(ref.Y18.values, y_pred)

#%%
ref['Y18'] = y_pred
ref.to_csv('submit/submit_19.csv',index=False)
