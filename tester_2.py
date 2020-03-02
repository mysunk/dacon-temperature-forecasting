# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 20:07:42 2020

@author: guseh
"""

from sklearn.linear_model import ElasticNet
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np
from util import *
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from functools import partial

def objective(params, train_set, n_folds=10,seed=0):
    model = ElasticNet(**params)
    score= make_scorer(mse_AIFrenz, greater_is_better=True)
    cv_results = cross_val_score(model, train_set[0], train_set[1], cv=n_folds,n_jobs=-1, verbose=0, scoring=score)
    cv_loss = np.mean(cv_results)
    # Dictionary with information for evaluation
    return {'loss': cv_loss, 'params': params, 'status': STATUS_OK}

# load raw data
train_1, train_2, train_label_1, train_label_2, test, sample = load_dataset('data_raw/')

train_1['time'] = train_1['id'] % 144
train_2['time'] = train_2['id'] % 144

# Elastic net
space_eln = {'max_iter': hp.choice('max_iter',np.arange(1000,10000,step=100,dtype=int)),
             "alpha": hp.loguniform('alpha',np.log(0.0001),np.log(1000)),
             'l1_ratio':hp.choice('l1_ratio',np.arange(0.0, 1.0, 0.1)),
             'random_state' : 1
             }

space_eln_2 = {'max_iter': hp.choice('max_iter',np.arange(1000,10000,step=100,dtype=int)),
             "alpha": hp.loguniform('alpha',np.log(0.2),np.log(0.4)),
             'l1_ratio':hp.choice('l1_ratio',np.arange(0.7, 0.9, 0.01)),
             'random_state' : 1
             }

tpe_algorithm = tpe.suggest  # Algorithm
bayes_trials = Trials()  # Trials object to track progress

train_set = (train_1, train_label_1)
fmin_objective = partial(objective, train_set=train_set,n_folds=5)
best = fmin(fn=fmin_objective, space=space_eln, algo=tpe_algorithm, max_evals=50, trials=bayes_trials)
