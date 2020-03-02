# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 20:07:42 2020

@author: guseh
"""

from sklearn.linear_model import ElasticNet
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import numpy as np
from util import *

# load raw data
train_1, train_2, train_label_1, train_label_2, test, sample = load_dataset('data_raw/')
    
# 하루 단위로 데이터 샘플 만듦
train_1 = make_day_sample(train_1)
train_2 = make_day_sample(train_2)
train_label_1 = make_day_label(train_label_1)
train_label_2 = make_day_label(train_label_2)

# Elastic net
regr = ElasticNet(random_state=0)
parametersGrid = {"max_iter": [1000],
                  "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                  "l1_ratio": np.arange(0.0, 1.0, 0.1)}
# for 0th model
score= make_scorer(mse_AIFrenz, greater_is_better=False)
grid = GridSearchCV(regr, parametersGrid, scoring=score, cv=5,n_jobs=-1)
grid.fit(train_1, train_label_1[0])