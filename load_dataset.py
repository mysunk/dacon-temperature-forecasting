# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 20:28:31 2020

@author: guseh
"""

import pandas as pd
import numpy as np

# load raw dataset
train = pd.read_csv('data_raw/train.csv')
test = pd.read_csv('data_raw/test.csv')

# preprocessing
# train = train.drop(columns = ['X11','X14','X16','X19','X34'])
test = test.drop(columns = ['X11','X14','X16','X19','X34'])

# split data and label
train_1 = train.loc[:24*6*30-1,'X00':'X39']
train_2 = train.loc[24*6*30:,'X00':'X39']

train_label_1_ref = train.loc[:24*6*30-1,'Y00':'Y17']
train_label_1 = pd.read_csv('data_npy/Y_18.csv')
train_label_2 = train.loc[24*6*30:,'Y18']

# preprocessing
# train_1['time'] = train_1['id'] % 144
# train_2['time'] = train_2['id'] % 144
# train_1['day'] = (train_1['id'] / 144).astype(int)
# train_2['day'] = (train_2['id'] / 144).astype(int)
# test['time'] = test['id'] % 144
# test['day'] = (test['id'] / 144).astype(int)
# train_1 = train_1.drop(columns='id')
# train_2 = train_2.drop(columns='id')
train_2 = train_2.reset_index(drop=True)
train_label_2 = train_label_2.reset_index(drop=True)

#%% Make profile -- 하루단위
profiles = []
profile = pd.DataFrame(train_1.loc[:,['X00']]) # 기온
profile = np.reshape(profile.values, (-1,6*24))

profiles.append(profile)
profile = pd.DataFrame(train_1.loc[:,['X01']]) # 현지기압
profile = np.reshape(profile.values, (-1,6*24))
profiles.append(profile)
profile = np.concatenate(profiles,axis=1)

for i in range(1,30):
    new_data = pd.concat([train_1.iloc[i*144:(i+1)*144].reset_index(drop=True), pd.DataFrame(np.tile(profile[i-1,:], (144, 1)))],axis=1)
    if i==1:
        train_1_p = new_data
    else:
        train_1_p = pd.concat([train_1_p,new_data],axis=0)
    


#%% For setting
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
# get_ipython().run_line_magic('matplotlib', 'inline')

font = {'family' : 'Helvetica',
        'size'   : 11}

#%% save
train_1.to_csv('data_npy/train_1.csv',index=False)

train_2.to_csv('data_npy/train_2.csv',index=False)

train_label_1.to_csv('data_npy/train_label_1.csv',index=False)

train_label_2.to_csv('data_npy/train_label_2.csv',index=False)

test.to_csv('data_npy/test.csv',index=False)

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

#%%
import matplotlib.pyplot as plt
plt.plot(train_label_1)
plt.plot(train_label_1_ref.loc[:,'Y16'])
plt.plot(train_label_1_ref.loc[:,'Y17'])

plt.legend(['sw','Y16','Y17'])

#%%
plt.figure
plt.plot(train.loc[:,'X00'])

#%%
plt.figure
plt.plot(test.loc[:,'X00'])