# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:19:59 2020

@author: guseh
"""
from util import *
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
# clf
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

def lgb_param():
    lgb_param = {
        'bagging_freq' :            2,
        'boosting' :                'gbdt',
        'colsample_bynode' :        0.5614113707540148,
        'colsample_bytree' :        0.5236228311328034,
        'learning_rate' :           0.11074930666097238,
       ' max_bin' :                 100,
        'max_depth' :               10,
        'min_child_weight' :        4,
        'min_data_in_leaf' :        30,
        'num_leaves' :              20,
        'reg_alpha' :               0.0346297125460458,
        'reg_lambda':               3.641221146386232,
        'subsample':                0.891312377550311,
        'tree_learner':             'feature',
        'random_state':             0,
        'n_jobs':                   -1,
    }
    return lgb_param

def eln_param():
    eln_param =  {
        'max_iter':                 10,
         "alpha":                   1,
         'l1_ratio':                0.1,
         'random_state' :           0,
         }

def rf_param(self):
    rf_param =  {
        'max_depth':                10,
        'max_features':             10,
        'n_estimators':             100,
        #'criterion':               hp.choice('criterion', ["gini", "entropy"]),
        'random_state' :            0,
       }

def svr_param(self):
    svr_param = {
        'kernel':                   'linear',
        'C':                        1,
        'gamma':                    1e-3,
        'epsilon':                  0.2,
        'random_state' :           0,
        }

# def tmps():
if __name__ == '__main__':
    # load cv result
    param = lgb_param() # pre-defined param
    trials = load_obj('lgb_0320_3')
    # param = trials.best_trial['result']['params']
    # param = trials.results['result']['params']
    param =trials[30]['params'] # 68, 30, 67
     ####'lgb_0320_3 의 30번째, 3000번 -- 2.67나옴
    # load dataset
    train = pd.read_csv('data_raw/train.csv')
    test = pd.read_csv('data_raw/test.csv')
    train_label = pd.read_csv('data_npy/Y_18.csv')
    
    # split data and label
    train = train.loc[:,'id':'X39']
    train['time'] = train.id.values % 144
    train = train.drop(columns = 'id')
    test = test.loc[:,'id':'X39']
    test['time'] = test.id.values % 144
    test = test.drop(columns = 'id')
    
    # declare dataset
    N = 3
    train_label = train_label[N:]
    train_partial = train.iloc[-N:,:] # 뒤의 N개 잘라내서 저장
    train = add_profile_v2(train, ['X00', 'X12','X11'],N) # 기온만 추가
    train = train.drop(columns = 'index')
    test = pd.concat([train_partial, test], axis=0).reset_index(drop=True)
    
    test = add_profile_v2(test,['X00', 'X12','X11'],N)
    test = test.drop(columns = 'index')
    # First train phase
    dtrain = lgb.Dataset(train, label=train_label)
    model = lgb.train(param, train_set = dtrain,valid_sets = [dtrain], num_boost_round=3000,verbose_eval=True,
                                 feval = mse, early_stopping_rounds=10)
    
    y_pred = model.predict(test)
    print('diff with sw is',mse_AIFrenz(ref.Y18.values, y_pred))
    
    """ other classifier
    # rf
    model = RandomForestRegressor(**param)
    model.fit(train, train_label)
    """