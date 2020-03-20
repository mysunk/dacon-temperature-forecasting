# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 23:05:05 2020

@author: guseh
"""
import lightgbm as lgb
# import xgboost as xgb
# import catboost as ctb
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials, hp
import numpy as np
from util import *
from sklearn.model_selection import KFold


def lgb_space(fit_params):
    # LightGBM parameters
    lgb_param = {
        'learning_rate':            hp.uniform('learning_rate',    0.05, 0.3),
        'max_depth':                hp.quniform('max_depth',        10, 200, 1),
        'num_leaves':               hp.quniform('num_leaves',       2, 30, 1),
        'min_data_in_leaf':		    hp.quniform('min_data_in_leaf',	10, 300, 10),	
        'reg_alpha':                hp.uniform('reg_alpha',0.0,100.0),
        'reg_lambda':               hp.uniform('reg_lambda',0.0,100.0),
        'min_child_weight':         hp.quniform('min_child_weight', 1, 10, 1),
        'colsample_bytree':         hp.uniform('colsample_bytree', 0.1, 1.0),
        'colsample_bynode':		    hp.uniform('colsample_bynode',0.1,1.0),
        'bagging_freq':			    hp.quniform('bagging_freq',	1,10,1),
        'tree_learner':			    hp.choice('tree_learner',	['serial','feature','data','voting']),
        'subsample':                hp.uniform('subsample', 0.1, 1),
        'boosting':			        hp.choice('boosting', ['gbdt','rf']),
        'max_bin':			        hp.quniform('max_bin',		10,300,10),
        'random_state':             fit_params['seed'],
        'n_jobs':                   -1,
    }
    return lgb_para

def xgb_space(self):
    return None

def ctb_space(self):
    return None

class HyperOptimize(object):

    def __init__(self, train, train_label, nfold):
        self.train = train
        self.train_label  = train_label
        self.nfold = nfold
        
        # optimize cv loss
    def process(self, fn_name, space, trials, algo, max_evals):
        fn = getattr(self, fn_name)
        try:
            result = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result, trials
    
    def lgb_objective(self, para):
        para['reg_params'] = make_param_int(para['reg_params'], ['max_depth','num_leaves','min_data_in_leaf',
                                     'min_child_weight','bagging_freq','max_bin'])
        losses = []
        kf = KFold(n_splits=self.nfold, random_state=None, shuffle=False)
        for i, (train_index, test_index) in enumerate(kf.split(self.train, self.train_label)):
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
            model = lgb.train(para['reg_params'], train_set = dtrain,  
                              valid_sets=[dtrain, dvalid], **para['fit_params'])
            # pred = model.predict(dvalid)
            # loss = self.loss_func(y_test, pred)
            losses.append(model.best_score['valid_1']['mse_modified'])
        return {'loss': np.mean(losses,axis=0),'params':para ,'status': STATUS_OK}