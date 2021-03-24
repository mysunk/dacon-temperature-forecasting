# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 23:41:55 2020

@author: guseh
"""


trials = load_obj('tmp')
trials = sorted(trials, key=lambda k: k['loss'])
param = trials[0]['params']
train = load_dataset_v2('train',12, 20, True)
train = train.values
# train_label = train_label[label].values
_, train_label = load_dataset_v2('train2',12, 20, True)
train_label = np.hstack([np.load('data_pre/Y18_with_Y13.npy'),train_label.values])

test = load_dataset_v2('test',12, 20, True)

for seeds in range(10):
    param['random_state']=seeds
    dtrain = lgb.Dataset(train, label=train_label)
    model = lgb.train(param, train_set = dtrain,valid_sets = [dtrain], num_boost_round=1000,verbose_eval=True,
                                 early_stopping_rounds=10,feval=mse_AIFrenz_lgb)
    y_pred = model.predict(test)