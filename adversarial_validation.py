# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 20:17:16 2020

@author: guseh
"""
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import lightgbm as lgb
import numpy as np
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials, hp

train = pd.read_csv('data_npy/train_1.csv')
test = pd.read_csv('data_npy/train_2.csv')

# 하루 단위 profile로 바꿈

train = train.loc[:,['X00','X07','X28','X31','X32']]
test = test.loc[:,['X00','X07','X28','X31','X32']]
train = train.values
test = test.values
train = np.reshape(train,(30,-1)) # 하루 단위
test = np.reshape(test,(3,-1))
train = pd.DataFrame(train)
test = pd.DataFrame(test)


train['is_train'] = 1
test['is_train'] = 0

train_test = pd.concat([train, test], axis =0)
target = train_test['is_train'].values
train_test = train_test.drop(columns=['is_train'])

# lgb params
param = {
        'boosting': 'gbdt',
        'application': 'binary',
        'metric': 'auc', 
        'learning_rate': 0.1,
        'num_leaves': 32,
        'max_depth': 8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'feature_fraction': 0.7,
}

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=2)
oof = np.zeros(len(train_test))

for fold_, (trn_idx, val_idx) in enumerate(kf.split(train_test.values, target)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(train_test.iloc[trn_idx], label=target[trn_idx])
    val_data = lgb.Dataset(train_test.iloc[val_idx], label=target[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], 
                    verbose_eval=1, early_stopping_rounds = 100)
    oof[val_idx] = clf.predict(train_test.iloc[val_idx], num_iteration=clf.best_iteration)