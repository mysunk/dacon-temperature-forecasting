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
from sklearn.multioutput import MultiOutputRegressor

class Params(object):
    
    def __init__(self, config):
        self.param = dict()
        self.config = config


        self.param = lgb_para
    
    def get_param(self):
        return self.param

class HPOpt_cv(object):

    def __init__(self, train, train_label, config):
        self.train = train
        self.train_label  = train_label
        self.config = config

    def process(self, fn_name, space, trials, algo, max_evals):
        fn = getattr(self, fn_name)
        try:
            result = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result, trials

    def lgb_reg(self, para):
        reg = lgb.LGBMRegressor(**para['reg_params'])
        return self.train_reg(reg, para)

    def train_reg(self, reg, para):
        nfold = self.config['nfold']
        losses = []
        kf = KFold(n_splits=nfold, random_state=None, shuffle=False)
        model = MultiOutputRegressor(reg, n_jobs=-1)
        for i, (train_index, test_index) in enumerate(kf.split(self.train, self.train_label)):
                x_train = self.train.iloc[train_index]
                y_train = self.train_label.iloc[train_index]
                x_test = self.train.iloc[test_index]
                y_test = self.train_label.iloc[test_index]
                model.fit(x_train, y_train)
                pred = model.predict(x_test)
                loss = para['loss_func'](y_test, pred)
                losses.append(loss)
        return {'loss': np.mean(losses,axis=0), 'status': STATUS_OK}