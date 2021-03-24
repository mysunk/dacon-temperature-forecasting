# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 22:50:12 2020

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
        'max_depth':                2,
        'max_features':             9,
        'n_estimators':             500,
        'min_samples_leaf':         1,
        'min_samples_split':        6,
        'random_state' :            0,
       }
    return rf_param

label = 'Y13'
result = []
method = 'rf'
nfold = 0
# random_states = range(3)
for seeds in range(10):
    # user
    # trials = load_obj('tmp_2')
    # trials = sorted(trials, key=lambda k: k['loss'])
    # param = trials[0]['params']
    param = rf_param()
    param['random_state']=seeds
    # param['max_depth'] = depth
    
    
    train, train_label = load_dataset_v2('train2',12, 20, True)
    
    train_label = train_label.values - np.load('predictions/'+label+'_pred_3day_lgb.npy')
    train = train.values
    
    test, _ = load_dataset_v2('train1',12, 20, True)
    test = test.values
    
    val = np.zeros((432))
    preds = []
    if nfold==0:
        if method == 'lgb':
            dtrain = lgb.Dataset(train, label=train_label)
            model = lgb.train(param, train_set = dtrain,valid_sets = [dtrain], num_boost_round=50,
                              feval=mse_AIFrenz_lgb,verbose_eval=True)
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
                # param['metric']='l2'
                dtrain = lgb.Dataset(x_train, label=y_train)
                dvalid = lgb.Dataset(x_test, label=y_test)
                model = lgb.train(param, train_set = dtrain,valid_sets = [dtrain, dvalid], num_boost_round=1000,verbose_eval=True,
                                         early_stopping_rounds=10,feval=mse_AIFrenz_lgb)
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
y_pred = np.mean(result, axis=0)

#%%

_, train_label = load_dataset_v2('train1',12, 20, True)
train_label = train_label['Y13'].values + y_pred
np.save('../data_pre/Y18_with_Y13.npy', train_label)
