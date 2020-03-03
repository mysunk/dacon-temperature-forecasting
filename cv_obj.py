# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 22:44:55 2020

@author: guseh
"""
from sklearn.linear_model import ElasticNet
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np
from util import *
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, STATUS_FAIL
from functools import partial
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor



class HPOpt(object):
    
    def __init__(self, random_state, clf_name):
        self.random_state = random_state
        self.space = {}
        clf = getattr(self, clf_name)
        clf()
    
    # parameter setting
    def eln(self):
        self.space =  {'max_iter': hp.choice('max_iter',np.arange(1000,10000,step=100,dtype=int)),
             "alpha": hp.loguniform('alpha',np.log(0.0001),np.log(1000)),
             'l1_ratio':hp.choice('l1_ratio',np.arange(0.0, 1.0, 0.1)),
             'random_state' : self.random_state,
             }
    
    def rf(self):
        self.space =  {'max_depth': hp.choice('max_depth', range(10,200,10)),
           'max_features': hp.choice('max_features', range(10,40)),
           'n_estimators': hp.choice('n_estimators', range(100,1000,10)),
           # 'criterion': hp.choice('criterion', ["gini", "entropy"]),
           'random_state' : self.random_state,
           }
    
    def svr(self):
        self.space = {'kernel': hp.choice('kernel',['linear', 'rbf','poly']),
                      'C':hp.uniform('C',1,10),
                      'gamma': hp.loguniform('gamma',np.log(1e-7),np.log(1e-1)),
                      'epsilon': hp.uniform('epsilon',0.1,0.9),
            }
        
    # optimize
    def process(self, fn_name, train_set, nfolds, trials, algo, max_evals):
        fn = getattr(self, fn_name)
        fmin_objective = partial(fn, train_set=train_set,nfolds=nfolds)
        try:
            result = fmin(fn=fmin_objective, space=self.space, algo=algo, max_evals=max_evals, trials=trials)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result, trials
    
    # objective function
    def eln_cv(self, params, train_set, nfolds):
        model = ElasticNet(**params)
        score= make_scorer(mse_AIFrenz, greater_is_better=True)
        cv_results = cross_val_score(model, train_set[0], train_set[1], cv=nfolds,n_jobs=-1, verbose=0, scoring=score)
        cv_loss = np.mean(cv_results)
        # Dictionary with information for evaluation
        return {'loss': cv_loss, 'params': params, 'status': STATUS_OK}
    
    def rf_cv(self, params, train_set, nfolds):
        model = RandomForestRegressor(**params)
        score= make_scorer(mse_AIFrenz, greater_is_better=True)
        cv_results = cross_val_score(model, train_set[0], train_set[1], cv=nfolds,n_jobs=-1, verbose=0, scoring=score)
        cv_loss = np.mean(cv_results)
        # Dictionary with information for evaluation
        return {'loss': cv_loss, 'params': params, 'status': STATUS_OK}
    
    def svr_cv(self, params, train_set, nfolds):
        model = MultiOutputRegressor(SVR(**params), n_jobs=-1)
        score= make_scorer(mse_AIFrenz, greater_is_better=True)
        cv_results = cross_val_score(model, train_set[0], train_set[1], cv=nfolds,n_jobs=-1, verbose=0, scoring=score)
        cv_loss = np.mean(cv_results)
        # Dictionary with information for evaluation
        return {'loss': cv_loss, 'params': params, 'status': STATUS_OK}