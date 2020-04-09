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

label = 'Y15'
result = []
method = 'rf'
nfold = 0
# random_states = range(3)
for depth in [5]:
# for seeds in range(1):
    # user
    # trials = load_obj('Y15_residual')
    # param = trials[0]['params']
    param = rf_param()
    # param['random_state']=seeds
    param['max_depth'] = depth
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
    
# np.save('predictions/'+label+'_residual_80day'+method+'.npy',np.mean(result, axis=0))
    
#%% 3일치 결과
plt.plot(val)
plt.plot(train_label)

# plt.plot(np.array(result).T)
plt.plot(result[0])
plt.plot(result[1])
plt.plot(result[2])
#%% 최종 예측
label = 'Y15'
data = np.load('predictions/'+label+'_pred_80day_lgb.npy')
data =np.ravel( data + np.mean(result, axis=0))

ref = pd.read_csv('submit/submit_25.csv')
plt.plot(data)
plt.plot(ref.Y18.values)
plt.plot(ref_prev.Y18.values)

#%% mse
# ref_prev = pd.read_csv('submit/sample_submission_v40.csv')
ref_prev = pd.read_csv('submit/sample_submission_v40.csv')
mean_squared_error(ref.Y18.values, ref_prev.Y18.values)
mean_squared_error(ref.Y18.values, data)
mean_squared_error(ref.Y18.values, data1)
mean_squared_error(ref.Y18.values, data2)
mean_squared_error(ref.Y18.values, data3)

mean_squared_error(ref.Y18.values, y_pred)
mean_squared_error(np.load('predictions/Y18_with_Y12_80day.npy'), np.load('predictions/Y18_with_Y13_80day.npy'))

#%%
np.save(('predictions/Y18_with_Y15_80day.npy'),data)

#%% 최종 예측
label = 'Y15'
data = np.load('predictions/'+label+'_pred_80day_lgb.npy')
data =np.ravel( data + np.mean(result, axis=0))

ref = pd.read_csv('submit/submit_25.csv')
plt.plot(data)
plt.plot(ref.Y18.values)
plt.plot(ref_prev.Y18.values)

#%%
data1 = np.load('predictions/Y18_with_Y12_80day.npy')
data2 = np.load('predictions/Y18_with_Y13_80day.npy')
data3 = np.load('predictions/Y18_with_Y15_80day.npy')
plt.plot(ref_prev.Y18.values)
plt.plot(ref.Y18.values)
# plt.plot(y_pred)
plt.plot(data1)
plt.plot(data2)
plt.plot(data3)

mean_squared_error(ref.Y18.values, y_pred)

y_pred = data1*0.3 + data2 * 0.5 + data3 * 0.2

y_pred = data1 * 0.3 + ref.Y18.values * 0.7

#%%
for i in range(len(result)):
    print(mean_squared_error(np.ravel(np.load('predictions/Y15_pred_80day_lgb.npy')+result[i]),ref.Y18.values))
