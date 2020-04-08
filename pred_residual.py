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

label = 'Y12'
result = []
method = 'rf'
nfold = 0
# random_states = range(3)
# for depth in range(2,10):
for seeds in range(10):
    # user
    # trials = load_obj('tmp')
    # trials = sorted(trials, key=lambda k: k['loss'])
    # param = trials[0]['params']
    param = rf_param()
    param['random_state']=seeds
    # param['max_depth'] = depth
    # param['min_samples_split'] = depth
    
    
    train, train_label = load_dataset_v2('train2',12, 20, True)
    # del train['time']
    
    data = []
    # data.append(np.load('predictions/'+label+'_pred_3day_svr.npy'))
    #data.append(np.load('predictions/'+label+'_pred_3day_rf.npy'))
    data.append(np.load('predictions/'+label+'_pred_3day_lgb.npy'))
    
    data = np.mean(data, axis=0)
    train_label = train_label.values - data
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
Y01 = np.load('predictions/Y01_pred_80day_lgb.npy')
Y02 = np.load('predictions/Y02_pred_80day_lgb.npy')

Y01_res_lgb = np.load('predictions/Y01_residual_80day_lgb.npy')
Y02_res = np.load('predictions/Y02_residual_80day.npy')

Y18_01 = Y01 + np.mean(result, axis=0) #Y01_res
Y18_01_lgb = Y01 + Y01_res_lgb
Y18_02 = Y02 + Y02_res

ref_prev = pd.read_csv('submit/submit_11.csv')
# ref_bad = pd.read_csv('submit/submit_17.csv')
ref = pd.read_csv('submit/sample_submission_v39.csv')
plt.plot(ref.Y18.values)
plt.plot(ref_prev.Y18.values)
# plt.plot(ref_bad.Y18.values)
plt.plot(Y18_01)
plt.plot(Y18_01_lgb)
# plt.plot(y_pred)
# plt.plot(Y18_02)
plt.legend(['ref','ref_prev','proposed1','proposed2'])
y_pred = Y18_01 * 0.2 + ref.Y18.values * 0.8

#%%
Y13 = np.load('predictions/Y13_pred_80day_lgb.npy')
Y15 = np.load('predictions/Y15_pred_80day_lgb.npy')
# Y18_13 = Y13 + np.mean(result, axis=0)# result[2]# np.mean(result, axis=0) #Y01_res
Y18_15 = Y15 + result[0]# np.mean(result, axis=0)# result[2]# np.mean(result, axis=0) #Y01_res

ref_prev = pd.read_csv('submit/submit_11.csv')
ref = pd.read_csv('submit/sample_submission_v40.csv')
# plt.plot(Y18_15)
plt.plot(y_pred)
plt.plot(ref.Y18.values)
# plt.plot(tmp.Y18.values)
plt.plot(ref_prev.Y18.values)

#%%
Y12 = np.load('predictions/Y12_pred_80day_lgb.npy')

Y18_12 = Y12 + np.mean(result, axis=0)# result[0]# np.mean(result, axis=0)# result[2]# np.mean(result, axis=0) #Y01_res
ref = pd.read_csv('submit/sample_submission_v40.csv')
plt.plot(ref.Y18.values)
plt.plot(Y18_12)
#%%
# print(mean_squared_error(Y02 + np.mean(result, axis=0),ref_prev.Y18.values))
for i in range(len(result)):
    print(mean_squared_error(Y12 + result[i],ref.Y18.values))
    
#%%
y_pred = Y18_13 * 0.6 + Y18_15 * 0.4
for i in range(10):
    print(mean_squared_error(Y18_15 * 0.1*i + Y18_13 * (1-0.1*i),ref.Y18.values))

#%% ensemble
preds = []
preds.append(np.load('pred_1.csv.npy'))
preds.append(np.load('pred_2.csv.npy'))
preds.append(ref.Y18.values)
y_pred = np.dot(np.array([0.5, 0.5]),preds)
#%%
interv = range(5000,6000)
print(mean_squared_error(ref_prev.Y18.values, y_pred))

#%%
# plt.plot(preds[0])
# plt.plot(preds[1])
plt.plot(y_pred)
plt.plot(ref.Y18.values)
plt.plot(ref_prev.Y18.values)
plt.plot(ref_bad.Y18.values)
plt.legend(['pred','ref','ref_prev','ref_bad'])

#%%
ref = pd.read_csv('submit/sample_submission_v40.csv')
mean_squared_error(ref.Y18.values, y_pred)
y_pred = Y18_12 * 0.3 + ref.Y18.values * 0.7

#%%
ref['Y18'] = y_pred
ref.to_csv('submit/submit_19.csv',index=False)

#%%
data1 = np.load('predictions/Y13_pred_80day_lgb.npy')
data2 = np.load('predictions/Y15_pred_80day_lgb.npy')
residual1 = np.load('predictions/Y13_residual_80day.npy')
residual2 = np.load('predictions/Y15_residual_80day.npy')

ref_prev = pd.read_csv('submit/submit_11.csv')

# plt.plot(data1+residual1)
plt.plot(data2+residual2)
plt.plot(ref.Y18.values)
plt.plot(ref_prev.Y18.values)
plt.plot( (data1+residual1) * (0.3) + (data2+residual2) *(0.7))
plt.legend(['Y15','ref','ref_prev','ensemble'])

for i in range(10):
    y_pred = (data1+residual1) * (0.1*i) + (data2+residual2) *(1-0.1*i)
    print(mean_squared_error(ref.Y18.values, y_pred))

#%%
plt.plot( np.load('predictions/'+label+'_pred_80day_lgb.npy')+result[0])
plt.plot( np.load('predictions/'+label+'_pred_80day_lgb.npy')+result[1])
plt.plot( np.load('predictions/'+label+'_pred_80day_lgb.npy')+result[2])
plt.plot( np.load('predictions/'+label+'_pred_80day_lgb.npy')+result[3])
plt.plot( np.load('predictions/'+label+'_pred_80day_lgb.npy')+result[4])
plt.plot(ref.Y18.values)
plt.legend(['0','1','2','3','4','ref'])

mean_squared_error(ref.Y18.values, np.load('predictions/'+label+'_pred_80day_lgb.npy')+result[8])

y_pred = (data1+residual1) * (0.3) + (data2+residual2) *(0.7)

#%%
try_1 = pd.read_csv('submit/submit_22.csv')
try_2 = pd.read_csv('submit/submit_23.csv')
ref_1 = pd.read_csv('submit/sample_submission_v40.csv')
ref_2 = pd.read_csv('submit/sample_submission_v39.csv')
ref_3 = pd.read_csv('submit/submit_11.csv')

plt.plot(np.array([try_1.Y18.values,try_2.Y18.values,ref_1.Y18.values,ref_2.Y18.values,ref_3.Y18.values]).T)
plt.legend(['1','2','3','4','5'])