# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 20:28:31 2020

@author: guseh
"""

import pandas as pd

# load raw dataset
train = pd.read_csv('data_raw/train.csv')
test = pd.read_csv('data_raw/test.csv')

# split data and label
train_1 = train.iloc[:24*6*30,:41] # 30일간 X00 ~ X39
train_2 = train.iloc[24*6*30:,:41] # 3일간 X00 ~ X39
train_label_1 = train.iloc[:24*6*30,41:-1] # 30일간 Y00 ~ Y17
train_label_2 = train.iloc[24*6*30:,-1] # 3일간 Y18

# preprocessing
train_1['time'] = train_1['id'] % 144
train_2['time'] = train_2['id'] % 144
train_1['day'] = (train_1['id'] / 144).astype(int)
train_2['day'] = (train_2['id'] / 144).astype(int)
test['time'] = test['id'] % 144
test['day'] = (test['id'] / 144).astype(int)
train_1 = train_1.drop(columns='id')
train_2 = train_2.drop(columns='id')
train_2 = train_2.reset_index(drop=True)
train_label_2 = train_label_2.reset_index(drop=True)

# save
train_1.to_csv('data_npy/train_1.csv',index=False)
train_2.to_csv('data_npy/train_2.csv',index=False)
train_label_1.to_csv('data_npy/train_label_1.csv',index=False)
train_label_2.to_csv('data_npy/train_label_2.csv',index=False)
test.to_csv('data_npy/test.csv',index=False)

#%% For setting
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
# get_ipython().run_line_magic('matplotlib', 'inline')

font = {'family' : 'Helvetica',
        'size'   : 11}

#%% 
train_label_1_Y09 = train_label_1.loc[:,'Y09']
train_label_1_Y09.to_csv('data_npy/train_label_1_Y09.csv',index=False)

train_label_1_Y10 = train_label_1.loc[:,'Y10']
train_label_1_Y10.to_csv('data_npy/train_label_1_Y10.csv',index=False)

train_label_1_Y11 = train_label_1.loc[:,'Y11']
train_label_1_Y11.to_csv('data_npy/train_label_1_Y11.csv',index=False)

train_label_1_Y15 = train_label_1.loc[:,'Y15']
train_label_1_Y15.to_csv('data_npy/train_label_1_Y15.csv',index=False)

train_label_1_Y16 = train_label_1.loc[:,'Y16']
train_label_1_Y16.to_csv('data_npy/train_label_1_Y16.csv',index=False)

train_label_1_Y17 = train_label_1.loc[:,'Y17']
train_label_1_Y17.to_csv('data_npy/train_label_1_Y17.csv',index=False)