# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:19:59 2020

@author: guseh
"""
from util import *

# 학습데이터 로드


# load cv result
trials = load_obj('rf_10fold_1')
print(trials.best_trial['result']['params'])

# example
param = {'max_depth':150,
    'max_features':10,
    'n_estimators':100
    }

model = RandomForestRegressor(**param)

# 10-fold 학습
