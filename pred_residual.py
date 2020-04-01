# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 15:55:26 2020

@author: guseh
"""
from util import *
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

def svr_param():
    svr_param = {
        'kernel':                   'rbf',
        'C':                        1,
        'gamma':                    1,
        'epsilon':                  0.1,
        # 'random_state' :           0,
        }
    return svr_param

def rf_param():
    rf_param =  {
        'max_depth':                5,
        'max_features':             5,
        'n_estimators':             1000,
        #'criterion':               hp.choice('criterion', ["gini", "entropy"]),
        'random_state' :            0,
       }
    return rf_param

# user
label = 'Y13'
# trials= load_obj('0331/Y13_residual')
# param = trials[18]['params']
# param['max_depth'] = 8
param = rf_param()
method = 'rf'
nfold = 3
# param['max_depth']=1

train = pd.read_csv('data_pre/train_2.csv')
train = train.values
data = []
# data.append(np.load('data_pre/'+label+'_pred_3day_svr.npy'))
data.append(np.load('data_pre/'+label+'_pred_3day_rf.npy'))
# data.append(np.load('data_pre/'+label+'_pred_3day_lgb.npy'))

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

#%% analysis result
# y_pred = np.mean(preds,axis=0)
# ref = np.load('data_pre/Y13_80day_residual_rf.npy')
# for i in range(21):
#     plt.figure()
#      plt.plot(y_pred[range(500*i,500*i+500)])
#      plt.plot(ref[range(500*i,500*i+500)])
# np.save('data_pre/Y00_80day_residual_rf.npy',np.mean(preds,axis=0))

#%%
# plt.figure()
# plt.plot(train_label)
# plt.plot(val)
# plt.legend(['Y18-Y15_pred','residual_predict'])

    
    