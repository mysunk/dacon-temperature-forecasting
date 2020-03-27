# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:43:51 2020

@author: guseh
"""

import tuning
from util import *

# load dataset
train_1, train_2, train_label_1, train_label_2, test, sample = load_dataset('data_raw/')

Y16_pred = np.load('data_npy/Y16_pred.npy')

scaler = StandardScaler()
train.loc[:,:] = scaler.fit_transform(train.values)

