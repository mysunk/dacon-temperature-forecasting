# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 19:20:08 2020

@author: guseh
"""
from util import *
import matplotlib.pyplot as plt

#%% lightgbm
trials = load_obj('0326/Y09_1') # 1.8
trials = load_obj('0326/Y15_1') # 1.06
trials = load_obj('0326/Y16_1') # 1.48

#%% svr
trials = load_obj('0327/Y09') # 1.7
trials = load_obj('0327/Y15') # 1.04
trials = load_obj('0327/Y16') # 1.61

#%% rf
trials = load_obj('0328/Y16') # 1.37

#%% result -- 3일치 , svr
Y16_svr = np.load('data_pre/Y16_pred_3day_svr.npy')
Y09_svr = np.load('data_pre/Y09_pred_3day_svr.npy')
Y15_svr = np.load('data_pre/Y15_pred_3day_svr.npy')

#%% result -- 3일치 , lgb
Y16_lgb = np.load('data_pre/Y16_pred_3day_lgb.npy')
Y09_lgb = np.load('data_pre/Y09_pred_3day_lgb.npy')
Y15_lgb = np.load('data_pre/Y15_pred_3day_lgb.npy')
Y18 = np.load('data_pre/Y18.npy')

#%%
plt.plot(Y16_lgb,'--')
plt.plot(Y16_svr,'--')
plt.plot(Y18,'--')
plt.title('Y16')
plt.legend(['lgb','svr','Y18'])

#%%
label = 'Y09'
import numpy as np
import pandas as pd
# plt.plot(np.load('data_npy/Y16_pred_3day.npy'))
Y09 = np.load('data_npy/Y09_pred_3day.npy')
Y15 = np.load('data_npy/Y15_pred_3day.npy')
Y16 = np.load('data_npy/Y16_pred_3day.npy')
plt.plot(tmp)
plt.plot(Y18)
plt.legend(['pred','Y18'])
plt.title(label)

#%%
plt.plot(Y18 - tmp)
train = pd.read_csv('data_raw/train.csv')
train = train.iloc[4320:,:].reset_index(drop=True)
# plt.plot(irradiance_difference(train.X34.values))


#%%
Y18 = pd.read_csv('data_raw/train.csv')
Y18 = Y18.Y18.values[4320:]

#%%
train_label = pd.read_csv('data_raw/train.csv')
plt.plot(train_label.Y09.iloc[0:300])
plt.plot(train_label.Y16.iloc[0:300])
plt.plot(train_label.Y15.iloc[0:300])

#%%
plt.plot(Y09)
plt.plot(Y15)
plt.plot(Y16)
plt.plot(Y18)
plt.legend(['Y09','Y15','Y16','Y18'])

#%% replace outlier
Ys = np.zeros((432,4))
Ys[:,0] = Y09
Ys[:,1] = Y15
Ys[:,2] = Y16
Ys[:,3] = Y18

Ys = pd.DataFrame(Ys)
Ys.to_csv('matlab/3day_Y.csv',index=False)
#%%


#%%
Y15_res = pd.read_csv('matlab/res_15_2.csv',header=None)
y_pred_15 = np.load('data_npy/Y15_pred_30day.npy')
y_pred_15 = y_pred_15 + 1.7+ np.ravel(Y15_res.values)*0
# 
# tmp = y_pred + 0.8
interv = range(5000,6000)
plt.plot(tmp[interv])
plt.plot(ref.Y18.values[interv])

#%%
# Y16_res = pd.read_csv('matlab/res_15_2.csv',header=None)
y_pred_16 = np.load('data_npy/Y16_pred_30day.npy')
y_pred_16 = y_pred_16 +1.7
# 
# tmp = y_pred + 0.8
interv = range(5000,6000)
plt.plot(tmp[interv])
plt.plot(ref.Y18.values[interv])
#%%
# Y16_res = pd.read_csv('matlab/res_15_2.csv',header=None)
y_pred_09 = np.load('data_npy/Y09_pred_30day.npy')
y_pred_09 = y_pred_09 +1.7
# 
# tmp = y_pred + 0.8
interv = range(6000,7000)
plt.plot(y_pred[interv])
plt.plot(ref.Y18.values[interv])
#%%
ref = pd.read_csv('submit/sample_submission_v33.csv')
ref_ms = pd.read_csv('submit/submit_6.csv')

mse_AIFrenz(ref.Y18.values,y_pred)


mean_squared_error(ref_ms.Y18.values,y_pred)

#%%
y_pred = y_pred_15 * 0.5 + y_pred_16 * 0.3 + y_pred_09 * 0.2
ref_ms.loc[:,'Y18'] = y_pred
ref_ms.to_csv('submit/submit_7.csv',index=False)
