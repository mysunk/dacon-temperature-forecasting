# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:19:59 2020

@author: guseh
"""
from util import *
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.preprocessing import StandardScaler

# clf
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.svm import SVR

def lgb_param():
    lgb_param = {
        'bagging_freq' :            13,
        'boosting' :                'gbdt',
        'colsample_bynode' :        0.340689944410523,
        'colsample_bytree' :        0.7597589379552556,
        'learning_rate' :           0.04329330711902006,
       ' max_bin' :                 47,
        'max_depth' :               -1,
        'min_child_weight' :        25,
        'min_data_in_leaf' :        53,
        'num_leaves' :              173,
        'reg_alpha' :               0.17942844400628155,
        'reg_lambda':               0.3059871058500314,
        'subsample':                0.9045116843922111,
        'tree_learner':             'voting',
        'random_state':             0,
        'n_jobs':                   -1,
        'min_sum_hessian_in_leaf':  8,
    }
    return lgb_param

def eln_param():
    eln_param =  {
        'max_iter':                 10,
         "alpha":                   1,
         'l1_ratio':                0.1,
         'random_state' :           0,
         }
    return eln_param

def rf_param():
    rf_param =  {
        'max_depth':                10,
        'max_features':             10,
        'n_estimators':             100,
        #'criterion':               hp.choice('criterion', ["gini", "entropy"]),
        'random_state' :            0,
       }
    return rf_param

def svr_param():
    svr_param = {
        'kernel':                   'rbf',
        'C':                        5.6,
        'gamma':                    0.035,
        'epsilon':                  0.23,
        'random_state' :           0,
        }
    return svr_param

if __name__ == '__main__':
    # load cv result
    # user
    preds_all_test1 = []
    preds_all_test2 = []
    loss_results = []
    for param_num in range(5):# Change parameter
        sensor = 'Y17'
        method = 'lgb'
        save = False
        random_seeds = list(range(10))
        trials = load_obj('0409/'+sensor+'_'+method)
        # if np.isnan(trials[0]['loss']): del trials[0]
        # param_num = range(5)
        nfold = 10
        test1_savename = 'predictions/'+sensor+'_pred_3day_'+method+'.npy'
        test2_savename = 'predictions/'+sensor+'_pred_80day_'+method+'.npy'

        for seeds in random_seeds:# Change random state
            
            param = trials[param_num]['params']
            # param = lgb_param()
            param['metric']='l2'
            if not method is 'svr': param['random_state'] = seeds
            
            # User
            # load dataset
            label = sensor
            train, train_label = load_dataset_v2('train1',12, 20, True)
            train = train.values
            train_label = train_label[label].values
            
            test1, _ = load_dataset_v2('train2',12, 20, True)
            test2 = load_dataset_v2('test',12, 20, True)
            
            #============================================= load & pre-processing ==================================================
            
            if nfold==0:
                dtrain = lgb.Dataset(train, label=train_label)
                model = lgb.train(param, train_set = dtrain,valid_sets = [dtrain], num_boost_round=1000,verbose_eval=True,
                                             early_stopping_rounds=10)
                y_pred = model.predict(test)
            else:
                losses = np.zeros((nfold,2)) # 0:train, 1:val
                preds_test1 = []
                preds_test2 = []
                models = []
                kf = KFold(n_splits=nfold, random_state=None, shuffle=True)
                for i, (train_index, test_index) in enumerate(kf.split(train, train_label)):
                    print(i,'th fold training')
                    # train test split
                    if isinstance(train, (np.ndarray, np.generic) ): # if numpy array
                        x_train = train[train_index]
                        y_train = train_label[train_index]
                        x_test = train[test_index]
                        y_test = train_label[test_index]
                    else: # if dataframe
                        x_train = train.iloc[train_index]
                        y_train = train_label.iloc[train_index]
                        x_test = train.iloc[test_index]
                        y_test = train_label.iloc[test_index]
                    
                    # w.r.t method
                    if method == 'lgb':
                        dtrain = lgb.Dataset(x_train, label=y_train)
                        dvalid = lgb.Dataset(x_test, label=y_test)
                        model = lgb.train(param, train_set = dtrain,valid_sets = [dtrain, dvalid], num_boost_round=1000,verbose_eval=True,
                                                 early_stopping_rounds=10)
                    elif method == 'svr':
                        if not i: del param['metric']
                        model = SVR(**param)
                        model.fit(x_train, y_train)
                    elif method == 'rf':
                        if not i: del param['metric']
                        model = RandomForestRegressor(n_jobs=-1,**param)
                        model.fit(x_train, y_train)
                    # for evaluation
                    train_pred = model.predict(x_train)
                    valid_pred = model.predict(x_test)
                    losses[i,0]= mean_squared_error(y_train, train_pred)
                    losses[i,1]= mean_squared_error(y_test, valid_pred)
                    # test
                    preds_test1.append(model.predict(test1))
                    preds_test2.append(model.predict(test2))
                # average k-fold results
                y_pred_test1 = np.mean(preds_test1, axis=0)
                y_pred_test2 = np.mean(preds_test2, axis=0)
            loss_results.append(losses)
            preds_all_test1.append(y_pred_test1)
            preds_all_test2.append(y_pred_test2)
        
    y_pred_test1 = np.mean(preds_all_test1,axis=0)
    y_pred_test2 = np.mean(preds_all_test2,axis=0)
    if save:
        np.save(test1_savename,y_pred_test1)
        np.save(test2_savename,y_pred_test2)

#%%
index = []
for i in range(len(loss_results)):
    print(np.mean(loss_results[i][1])<0.35)
    index.append(np.mean(loss_results[i][1])<0.35)

#%%
test1 = []
for i in index:
    if i:
        test1.append(y_pred_test1[i])

test2 = []
for i in index:
    if i:
        test2.append(y_pred_test2[i])

#%%
np.save(test1_savename,np.mean(test1,axis=0))
np.save(test2_savename,np.mean(test2,axis=0))
