# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 19:20:08 2020

@author: guseh
"""
from util import *
import matplotlib.pyplot as plt

#%% lightgbm
trials = load_obj('0326/Y09_1') # 1.8
param_sequence = [22]
trials = load_obj('0326/Y15_1') # 1.06
param_sequence = [28,37,46,26,27]
trials = load_obj('0326/Y16_1') # 1.48
param_sequence = [36,28,27]

#%% svr
trials = load_obj('0327/Y09') # 1.7
param_sequence = [43]
trials = load_obj('0327/Y15') # 1.04
param_sequence = [22,30,32,21,26]
trials = load_obj('0327/Y16') # 1.61
param_sequence = [32, 30]
#%% rf
trials = load_obj('0328/Y09') # 1.79
param_sequence = [89]
trials = load_obj('0328/Y15') # 0.83
param_sequence = [87]
trials = load_obj('0328/Y16') # 1.37
param_sequence = [19]

#%% result -- 3일치 , svr
Y16_svr = np.load('data_pre/Y16_pred_3day_svr.npy')
Y09_svr = np.load('data_pre/Y09_pred_3day_svr.npy')
Y15_svr = np.load('data_pre/Y15_pred_3day_svr.npy')

# result -- 3일치 , lgb
Y16_lgb = np.load('data_pre/Y16_pred_3day_lgb.npy')
Y09_lgb = np.load('data_pre/Y09_pred_3day_lgb.npy')
Y15_lgb = np.load('data_pre/Y15_pred_3day_lgb.npy')

# result -- 3일치 , rf
Y16_rf = np.load('data_pre/Y16_pred_3day_rf.npy')
Y09_rf = np.load('data_pre/Y09_pred_3day_rf.npy')
Y15_rf = np.load('data_pre/Y15_pred_3day_rf.npy')
Y18 = np.load('data_pre/Y18.npy')

#%% result -- 80일치
Y16_svr = np.load('data_pre/Y16_pred_80day_svr.npy')
Y09_svr = np.load('data_pre/Y09_pred_80day_svr.npy')
Y15_svr = np.load('data_pre/Y15_pred_80day_svr.npy')

# result -- 3일치 , lgb
Y16_lgb = np.load('data_pre/Y16_pred_80day_lgb.npy')
Y09_lgb = np.load('data_pre/Y09_pred_80day_lgb.npy')
Y15_lgb = np.load('data_pre/Y15_pred_80day_lgb.npy')

# result -- 3일치 , rf
Y16_rf = np.load('data_pre/Y16_pred_80day_rf.npy')
Y09_rf = np.load('data_pre/Y09_pred_80day_rf.npy')
Y15_rf = np.load('data_pre/Y15_pred_80day_rf.npy')
Y18 = np.load('data_pre/Y18.npy')

Y09_mean = np.mean([Y09_lgb,Y09_svr,Y09_rf],axis=0)
Y15_mean = np.mean([Y15_lgb,Y15_svr,Y15_rf],axis=0)
Y16_mean = np.mean([Y16_lgb,Y16_svr,Y16_rf],axis=0)

#%%
res_09 = pd.read_csv('matlab/res_09.csv',header=None)
res_09 = res_09.values
res_15 = pd.read_csv('matlab/res_15.csv',header=None)
res_15 = res_15.values
res_16 = pd.read_csv('matlab/res_16.csv',header=None)
res_16 = res_16.values
#%%
offset = np.ravel(res_09)
mean_squared_error(Y09_mean+offset, ref.Y18.values)

offset = np.ravel(res_15)
mean_squared_error(Y15_mean+offset, ref.Y18.values)

offset = np.ravel(res_16)
mean_squared_error(Y16_mean+offset, ref.Y18.values)

y_pred = (Y09_mean+np.ravel(res_15)) * 0.1 + (Y15_mean+np.ravel(res_15)) * 0.4 + (Y16_mean+np.ravel(res_15)) * 0.5
# y_pred = (Y09_mean+0.6) * 0.1 + (Y15_mean+2.7) * 0.4 + (Y16_mean+0.4) * 0.5
# y_pred = (Y09_mean+0.6) * 0.1 + (Y15_mean+2.7) * 0.7 + (Y16_mean+0.4) * 0.2
# y_pred = (Y09_mean+0.8) * 0.1 + (Y15_mean+1.9) * 0.4 + (Y16_mean+0.6) * 0.5
mean_squared_error(y_pred , ref.Y18.values)
# mean_squared_error(ref_ms_prev.Y18.values , ref_ms.Y18.values)


#%%
ref = pd.read_csv('submit/sample_submission_v35.csv')
ref_ms = pd.read_csv('submit/submit_7.csv')
ref_ms_prev = pd.read_csv('submit/submit_6.csv')
ref_ms.Y18 = y_pred
ref_ms.to_csv('submit/submit_9.csv',index=False)

#%%
ref_ms_1 = pd.read_csv('submit/submit_8.csv')
ref_ms_2 = pd.read_csv('submit/submit_9.csv')
#%%
plt.plot(ref_ms_1.Y18.values[range(1500,2000)])
plt.plot(ref_ms_2.Y18.values[range(1500,2000)])
#%%
plt.plot(y_pred[range(500,1000)])
plt.plot(ref.Y18.values[range(500,1000)])

#%% Y09
plt.figure()
plt.plot(Y09_lgb,'--')
plt.plot(Y09_svr,'--')
plt.plot(Y09_rf,'--')
plt.plot(Y18)
plt.title('Y09')
plt.legend(['lgb','svr','rf','Y18'])

# Y15
plt.figure()
plt.plot(Y15_lgb,'--')
plt.plot(Y15_svr,'--')
plt.plot(Y15_rf,'--')
plt.plot(Y18)
plt.title('Y15')
plt.legend(['lgb','svr','rf','Y18'])

# Y16
plt.figure()
plt.plot(Y16_lgb,'--')
plt.plot(Y16_svr,'--')
plt.plot(Y16_rf,'--')
plt.plot(Y18)
plt.title('Y16')
plt.legend(['lgb','svr','rf','Y18'])

#%%


#%%
tmp = pd.DataFrame({'Y09_lgb': Y09_lgb,'Y09_svr':Y09_svr,'Y09_rf':Y09_rf})
tmp.to_csv('matlab/Y09.csv',index=False)
tmp = pd.DataFrame({'Y15_lgb': Y15_lgb,'Y15_svr':Y15_svr,'Y15_rf':Y15_rf})
tmp.to_csv('matlab/Y15.csv',index=False)
tmp = pd.DataFrame({'Y16_lgb': Y16_lgb,'Y16_svr':Y16_svr,'Y16_rf':Y16_rf})
tmp.to_csv('matlab/Y16.csv',index=False)


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
ref_ms.to_csv('submit/submit_8.csv',index=False)

#%%
ref = pd.read_csv('submit/sample_submission_v35.csv')
ref_ms = pd.read_csv('submit/submit_7.csv')
#%%

