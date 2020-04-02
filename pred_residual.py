# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 15:55:26 2020

@author: guseh
"""
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
        'n_estimators':             1000,
        #'criterion':               hp.choice('criterion', ["gini", "entropy"]),
        'random_state' :            0,
       }
    return rf_param

for label in ['Y13', 'Y15']:
    # user
    param = rf_param()
    if label == 'Y15': param['max_depth'] = 2
    method = 'rf'
    nfold = 0
    
    train = pd.read_csv('data_pre/train_2.csv')
    train = train.values
    data = []
    data.append(np.load('data_pre/'+label+'_pred_3day_svr.npy'))
    data.append(np.load('data_pre/'+label+'_pred_3day_rf.npy'))
    data.append(np.load('data_pre/'+label+'_pred_3day_lgb.npy'))
    
    Y18 = np.load('data_pre/Y18.npy')
    data = np.mean(data,axis=0)
    
    train_label = Y18 - data
    test = pd.read_csv('data_pre/test.csv')
    
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

#%% 3일치 결과
plt.plot(val)
plt.plot(train_label)

#%% 최종 예측
res = np.mean(preds,axis=0)
svr = np.load('data_pre/'+label+'_pred_80day_svr.npy')
lgb = np.load('data_pre/'+label+'_pred_80day_lgb.npy')
rf = np.load('data_pre/'+label+'_pred_80day_rf.npy')
ref = pd.read_csv('submit/sample_submission_v40.csv')
ref = ref.Y18.values

data = [svr, rf, lgb]
data = np.mean(data, axis=0)

summed = data + np.mean(preds,axis=0)
np.save('data_pre/Y18_pred_'+label+'.npy',summed)

#%% ensemble
preds = []
preds.append(np.load('data_pre/Y18_pred_Y13.npy'))
preds.append(np.load('data_pre/Y18_pred_Y15.npy'))
y_pred = np.dot(np.array([0.5, 0.5]),preds)

#%%
"""
ref = pd.read_csv('submit/submit_11.csv')
ref = pd.read_csv('submit/sample_submission_v40.csv')
plt.plot(ref.Y18.values[range(1000,2000)])
plt.plot(y_pred[range(1000,2000)])
mean_squared_error(ref.Y18.values, y_pred)
"""