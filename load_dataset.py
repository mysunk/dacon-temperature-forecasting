# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 20:28:31 2020

@author: guseh
"""

import pandas as pd
import numpy as np
from util import *
from sklearn.preprocessing import StandardScaler


train = pd.read_csv('data_raw/train.csv')
test = pd.read_csv('data_raw/test.csv')

train_label_1 = train.loc[:4319,'Y00':'Y17']
train_label_2 = train.loc[4320:4751,'Y18']

# add and delete feature
train = train.loc[:,'id':'X39']
test = test.loc[:,'id':'X39'].reset_index(drop=True)

X = pd.concat([train,test],axis=0).reset_index(drop=True) # 합침
X = process_dataset(X)
train_1 = X.iloc[:4320,:]
train_2 = X.iloc[4320:4752,:]
test = X.iloc[4752:,:]

train_1.to_csv('data_pre/train_1.csv',index=False)
train_2.to_csv('data_pre/train_2.csv',index=False)
test.to_csv('data_pre/test.csv',index=False)

train_label_1.to_csv('data_pre/train_label_1.csv',index=False)
train_label_2.to_csv('data_pre/train_label_2.csv',index=False)

#%%

train = pd.read_csv('data_raw/train.csv')
#train_label = pd.read_csv('data_npy/Y_18_trial_1.csv')

# split data and label
train_1, train_2, train_label_1, train_label_2, test, sample = load_dataset('data_raw/')
train = train_1
train_label = train_label_1.loc[:,args.label]
# train_label = Y18_ms.mean(axis=1)

# add and delete feature
train = train.loc[:,'id':'X39']
train['time'] = train.id.values % 144
train['X11_diff'] = irradiance_difference(train.X11.values) # 누적을 difference로 바꿈
train['X34_diff'] = irradiance_difference(train.X34.values)

N_T = args.N_T
N_S = args.N_S
train = train.loc[:,['time','X00','X07','X30','X31','X34','X34_diff']]
train = add_profile_v4(train, 'X31',N_T) # 온도
train = add_profile_v4(train, 'X34_diff',N_S) # 일사량

# match scale
scaler = StandardScaler()
train.loc[:,:] = scaler.fit_transform(train.values)


#%%
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'qt')

#%%
tmp = irradiance_difference(train.X34.values)
tmp = np.reshape(tmp, (-1,144))
