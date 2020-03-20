# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:42:18 2020

@author: guseh
"""

import numpy as np
from util import *

# load dataset

train = np.load('data_npy/train_pred_2.npy')
train = train[:,0]# .reshape(-1,1)
_, _, _, train_label, _, _ = load_dataset('data_raw/')
train_label = train_label.drop(columns='id')

model = AR(train)
model_fit = model.fit()