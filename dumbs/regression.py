# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:35:53 2020

@author: mskim
"""

import lightgbm as lgb
from sklearn.model_selection import KFold
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials, hp
import numpy as np
from util import *


def lgb_params(params):
    # LightGBM parameters
    lgb_reg_params = {
        'learning_rate':            params['reg_params']['learning_rate'],
        'max_depth':                params['reg_params']['max_depth'],
        'num_leaves':               params['reg_params']['num_leaves'],
        'min_data_in_leaf':		    params['reg_params']['min_data_in_leaf'],
        'reg_alpha':                params['reg_params']['reg_alpha'],
        'reg_lambda':               params['reg_params']['reg_lambda'],
        'min_child_weight':         params['reg_params']['min_child_weight'],
        'colsample_bytree':         params['reg_params']['colsample_bytree'],
        'colsample_bynode':		    params['reg_params']['colsample_bynode'],
        'bagging_freq':			    params['reg_params']['bagging_freq'],
        'tree_learner':			    params['reg_params']['tree_learner'],
        'subsample':                params['reg_params']['subsample'],
        'boosting':			        params['reg_params']['boosting'],
        'max_bin':			        params['reg_params']['max_bin'],
        'random_state':             params['fit_params']['seed'],
        'n_jobs':                   -1,
    }
    lgb_fit_params = {
        'feval':                    mse_AIFrenz_lgb,
        'num_boost_round':          params['fit_params']['num_boost_round'],
        'early_stopping_rounds':    params['fit_params']['early_stopping_rounds'], 
        'verbose_eval':             params['fit_params']['verbose_eval'],
    }
    lgb_para = dict()
    lgb_para['reg_params'] = lgb_reg_params
    lgb_para['fit_params'] = lgb_fit_params
    lgb_para['loss_func' ] = mse_AIFrenz_lgb
    return lgb_para

class lgb_net(object):

    def __init__(self, train, train_label, nfold, param):
        self.train = train
        self.train_label  = train_label
        self.nfold = nfold
        self.models = []
        self.param = param
        
    
    def fit(self):
        kf = KFold(n_splits=self.nfold, random_state=None, shuffle=False)
        for i, (train_index, test_index) in enumerate(kf.split(self.train, self.train_label)):
            print(i,'th fold training')
            if isinstance(self.train, (np.ndarray, np.generic) ): # if numpy array
                x_train = self.train[train_index]
                y_train = self.train_label[train_index]
                x_test = self.train[test_index]
                y_test = self.train_label[test_index]
            else: # if dataframe
                x_train = self.train.iloc[train_index]
                y_train = self.train_label.iloc[train_index]
                x_test = self.train.iloc[test_index]
                y_test = self.train_label.iloc[test_index]
            dtrain = lgb.Dataset(x_train, label=y_train)
            dvalid = lgb.Dataset(x_test, label=y_test)
            model = lgb.train(self.param['reg_params'], train_set = dtrain,  
                              valid_sets=[dtrain, dvalid], **self.param['fit_params'])
            self.models.append(model)
    
    def fit_predict(self):
        losses = []
        pred = np.zeros(self.train_label.shape)
        kf = KFold(n_splits=self.nfold, random_state=None, shuffle=False)
        for i, (train_index, test_index) in enumerate(kf.split(self.train, self.train_label)):
            print(i,'th fold training')
            if isinstance(self.train, (np.ndarray, np.generic) ): # if numpy array
                x_train = self.train[train_index]
                y_train = self.train_label[train_index]
                x_test = self.train[test_index]
                y_test = self.train_label[test_index]
            else: # if dataframe
                x_train = self.train.iloc[train_index]
                y_train = self.train_label.iloc[train_index]
                x_test = self.train.iloc[test_index]
                y_test = self.train_label.iloc[test_index]
            dtrain = lgb.Dataset(x_train, label=y_train)
            dvalid = lgb.Dataset(x_test, label=y_test)
            model = lgb.train(self.param['reg_params'], train_set = dtrain,  
                              valid_sets=[dtrain, dvalid], **self.param['fit_params'])
            self.models.append(model)
            # print(model.predict(x_test).shape)
            pred[test_index,0]= model.predict(x_test)
            
        return pred
    
    def predict(self, test): # for new test samples
        preds = []
        for i in range(self.nfold):
            preds.append(self.models[i].predict(test))
        return np.mean(preds,axis=0)