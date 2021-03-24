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
        'max_depth':                10,
        'max_features':             9,
        'n_estimators':             500,
        'min_samples_leaf':         1,
        'min_samples_split':        6,
        'random_state' :            0,
       }
    return rf_param
# for label in ['Y09','Y11','Y12','Y13','Y15','Y16','Y17']:
for label in ['Y17']:
    result = []
    method = 'rf'
    nfold = 10
    # for depth in range(1,5):
    for seeds in range(1):
        # user
        trials = load_obj('residual/'+label+'_residual_'+method)
        param = trials[0]['params']
        # param = rf_param()
        # param['random_state']=seeds
        param['max_depth'] = 2
        # param['min_samples_split'] = depth
        
        
        train, train_label = load_dataset_v2('train2',12, 20, True)
        # del train['time']
        
        data = []
        # data.append(np.load('predictions/'+label+'_pred_3day_svr.npy'))
        #data.append(np.load('predictions/'+label+'_pred_3day_rf.npy'))
        data.append(np.load('predictions/'+label+'_pred_3day_lgb.npy'))
        
        data = np.mean(data, axis=0)
        train_label = np.ravel(train_label.values - data)
        train = train.values
        
        test = load_dataset_v2('test',12, 20, True)
        test = test.values
        
        val = np.zeros((432))
        preds = []
        if nfold==0:
            if method == 'lgb':
                dtrain = lgb.Dataset(train, label=train_label)
                model = lgb.train(param, train_set = dtrain,valid_sets = [dtrain], num_boost_round=1000,early_stopping_rounds=10,
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
        plt.figure()
        plt.plot(val)
    np.save('residual_0411_modified/'+label+'_residual_3day'+method+'.npy',val)
    np.save('residual_0411_modified/'+label+'_residual_80day'+method+'.npy',np.mean(result, axis=0))
   
    
#%% 3일치 결과
plt.figure()
plt.plot(val)
plt.plot(train_label)

#%% 80일치
plt.plot(result[0])
plt.plot(result[1])
plt.plot(result[2])
