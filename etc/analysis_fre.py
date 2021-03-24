# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 19:20:08 2020

@author: guseh
"""
from util import *
import matplotlib.pyplot as plt
import numpy as np
#%% lightgbm tuning result
trials = load_obj('0326/Y09_1') # 1.8
param_sequence = [22]
trials = load_obj('0326/Y15_1') # 1.06
param_sequence = [28,37,46,26,27]
trials = load_obj('0326/Y16_1') # 1.48
param_sequence = [36,28,27]
trials = load_obj('0329/Y13_lgb') # 0.8
param_sequence = [47]

#%% svr tuning result
trials = load_obj('0327/Y09') # 1.7
param_sequence = [43]
trials = load_obj('0327/Y15') # 1.04
param_sequence = [22,30,32,21,26]
trials = load_obj('0327/Y16') # 1.61
param_sequence = [32, 30]
trials = load_obj('0329/Y13_svr') # 0.76
param_sequence = [25]
#%% rf tuning result
trials = load_obj('0328/Y09') # 1.79
param_sequence = [89]
trials = load_obj('0328/Y15') # 0.83
param_sequence = [87]
trials = load_obj('0328/Y16') # 1.37
param_sequence = [19]
trials = load_obj('0329/Y13_rf') # 0.75
param_sequence = [30]

#%%
trials = load_obj('0329/Y12_lgb')
#%% 0330 results
trials = load_obj('0330/Y00_rf') # 0.05
trials = sorted(trials, key=lambda k: k['loss'])
trials = load_obj('0330/Y00_svr') # 0.02
trials = sorted(trials, key=lambda k: k['loss'])
trials = load_obj('0330/Y00_lgb') # 0.22
trials = sorted(trials, key=lambda k: k['loss'])

trials = load_obj('0330/Y01_rf') # 0.86
trials = sorted(trials, key=lambda k: k['loss'])
trials = load_obj('0330/Y01_svr') # 0.91
trials = sorted(trials, key=lambda k: k['loss'])
trials = load_obj('0330/Y01_lgb') # 1.02
trials = sorted(trials, key=lambda k: k['loss'])

trials = load_obj('0330/Y02_rf') # 0.9
trials = sorted(trials, key=lambda k: k['loss'])
trials = load_obj('0330/Y02_svr') # 0.88
trials = sorted(trials, key=lambda k: k['loss'])
trials = load_obj('0330/Y02_lgb') # 0.99
trials = sorted(trials, key=lambda k: k['loss'])

trials = load_obj('0330/Y03_rf') # 0.62
trials = sorted(trials, key=lambda k: k['loss'])
trials = load_obj('0330/Y03_svr') # 0.51
trials = sorted(trials, key=lambda k: k['loss'])
trials = load_obj('0330/Y03_lgb') # 0.64
trials = sorted(trials, key=lambda k: k['loss'])

trials = load_obj('0330/Y04_rf') # 0.71
trials = sorted(trials, key=lambda k: k['loss'])
trials = load_obj('0330/Y04_svr') # 0.46
trials = sorted(trials, key=lambda k: k['loss'])
trials = load_obj('0330/Y04_lgb') # 0.71
trials = sorted(trials, key=lambda k: k['loss'])

#%%
trials = load_obj('0402/Y15_rf')
trials = load_obj('0402/Y15_lgb')
trials = load_obj('0402/Y13_svr')

#%% result -- 3일치
Y16_svr = np.load('../data_pre/Y16_pred_3day_svr.npy')
Y09_svr = np.load('../data_pre/Y09_pred_3day_svr.npy')
Y15_svr = np.load('../data_pre/Y15_pred_3day_svr.npy')
Y13_svr = np.load('../data_pre/Y13_pred_3day_svr.npy')

# result -- 3일치 , lgb
Y16_lgb = np.load('../data_pre/Y16_pred_3day_lgb.npy')
Y09_lgb = np.load('../data_pre/Y09_pred_3day_lgb.npy')
Y15_lgb = np.load('../data_pre/Y15_pred_3day_lgb.npy')
Y13_lgb = np.load('../data_pre/Y13_pred_3day_lgb.npy')

# result -- 3일치 , rf
Y16_rf = np.load('../data_pre/Y16_pred_3day_rf.npy')
Y09_rf = np.load('../data_pre/Y09_pred_3day_rf.npy')
Y15_rf = np.load('../data_pre/Y15_pred_3day_rf.npy')
Y13_rf = np.load('../data_pre/Y13_pred_3day_rf.npy')
Y18 = np.load('../data_pre/Y18.npy')

Y09_mean = np.mean([Y09_lgb,Y09_svr,Y09_rf],axis=0)
Y15_mean = np.mean([Y15_lgb,Y15_svr,Y15_rf],axis=0)
Y16_mean = np.mean([Y16_lgb,Y16_svr,Y16_rf],axis=0)
Y13_mean = np.mean([Y13_lgb,Y13_svr,Y13_rf],axis=0)

train = np.vstack([Y18 - Y09_mean,Y18 -Y13_mean,Y18 -Y15_mean,Y18 -Y16_mean]).T
np.save('3day_residual.npy',train)

#%% result -- 80일치
Y16_svr = np.load('../data_pre/Y16_pred_80day_svr.npy')
Y09_svr = np.load('../data_pre/Y09_pred_80day_svr.npy')
Y15_svr = np.load('../data_pre/Y15_pred_80day_svr.npy')
Y13_svr = np.load('../data_pre/Y13_pred_80day_svr.npy')
Y00_svr = np.load('../data_pre/Y00_pred_80day_svr.npy')

# result -- 
Y16_lgb = np.load('../data_pre/Y16_pred_80day_lgb.npy')
Y09_lgb = np.load('../data_pre/Y09_pred_80day_lgb.npy')
Y15_lgb = np.load('../data_pre/Y15_pred_80day_lgb.npy')
Y13_lgb = np.load('../data_pre/Y13_pred_80day_lgb.npy')

# result -- 
Y16_rf = np.load('../data_pre/Y16_pred_80day_rf.npy')
Y09_rf = np.load('../data_pre/Y09_pred_80day_rf.npy')
Y15_rf = np.load('../data_pre/Y15_pred_80day_rf.npy')
Y13_rf = np.load('../data_pre/Y13_pred_80day_rf.npy')
Y00_rf = np.load('../data_pre/Y00_pred_80day_rf.npy')
Y18 = np.load('../data_pre/Y18.npy')

Y00_mean = np.mean([Y00_svr,Y00_rf],axis=0)
Y15_mean = np.mean([Y15_lgb,Y15_svr,Y15_rf],axis=0)
Y16_mean = np.mean([Y16_lgb,Y16_svr,Y16_rf],axis=0)
Y13_mean = np.mean([Y13_lgb,Y13_svr,Y13_rf],axis=0)
"""
Y09_res_rf = np.load('data_pre/Y09_80day_residual_rf.npy')
Y13_res_rf = np.load('data_pre/Y13_80day_residual_rf.npy')
Y15_res_rf = np.load('data_pre/Y15_80day_residual_rf.npy')
Y16_res_rf = np.load('data_pre/Y16_80day_residual_rf.npy')

Y09_res_svr = np.load('data_pre/Y09_80day_residual_svr.npy')
Y13_res_svr= np.load('data_pre/Y13_80day_residual_svr.npy')
Y15_res_svr= np.load('data_pre/Y15_80day_residual_svr.npy')
Y16_res_svr= np.load('data_pre/Y16_80day_residual_svr.npy')

Y18_1 = Y09_mean + Y09_res_rf * 0.7 + Y09_res_svr * 0.3
Y18_2 = Y13_mean + Y13_res_rf * 0.7 + Y13_res_svr * 0.3
Y18_3 = Y15_mean + Y15_res_rf * 0.7 + Y15_res_svr* 0.3
Y18_4 = Y16_mean + Y16_res_rf * 0.7+ Y16_res_svr* 0.3
"""
Y13_res = np.load('../data_pre/Y13_80day_residual_rf.npy')
Y15_res = np.load('../data_pre/Y15_80day_residual_rf.npy')
Y16_res = np.load('../data_pre/Y16_80day_residual_rf.npy')
Y00_res = np.load('../data_pre/Y00_80day_residual_rf.npy')
# res_svr = np.load('data_pre/80day_residual_svr.npy')

Y18_00 = Y00_rf +  + Y00_res
Y18_13 = Y13_mean +  + Y13_res
Y18_15 = Y15_mean +  + Y15_res

plt.plot(Y18_00[range(2000,3000)])
plt.plot(Y18_13[range(2000,3000)])
plt.plot(Y18_15[range(2000,3000)])

#%%
ref = pd.read_csv('../submit/sample_submission_v40.csv')
ref = pd.read_csv('../submit/submit_11.csv')
mean_squared_error(Y18_00,ref.Y18.values)

mean_squared_error(Y18_00,Y18_15)

#%%
for i in range(20):
    plt.figure()
    interv = range(i*500,i*500+500)
    plt.plot(Y18_00[interv])
    plt.plot(Y18_13[interv])
    plt.plot(Y18_15[interv])
    plt.plot(ref.Y18.values[interv])

#%% plot residual
plt.plot(res_rf[:,1])
# plt.plot(res_svr[:,0])

mean_squared_error(res_rf[:,0],res_svr[:,0])
#%% Make result
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


#%% load reference
ref = pd.read_csv('../submit/sample_submission_v33.csv')
ref_ms = pd.read_csv('../submit/submit_7.csv')
ref_ms_prev = pd.read_csv('../submit/submit_6.csv')
ref_ms.Y18 = y_pred
ref_ms.to_csv('submit/submit_9.csv',index=False)

#%% plot result -- 80일
plt.plot(ref_ms_1.Y18.values[range(1500,2000)])
plt.plot(ref_ms_2.Y18.values[range(1500,2000)])=
plt.plot(y_pred[range(500,1000)])
plt.plot(ref.Y18.values[range(500,1000)])

#%% plot results -- 3일
plt.figure()
plt.plot(Y09_lgb,'--')
plt.plot(Y09_svr,'--')plt.plot(Y09_rf,'--')
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

# Y13
plt.figure()
plt.plot(Y13_lgb,'--')
plt.plot(Y13_svr,'--')
plt.plot(Y13_rf,'--')
plt.plot(Y18)
plt.title('Y13')
plt.legend(['lgb','svr','rf','Y18'])


#%% to matlab
tmp = pd.DataFrame({'Y09_lgb': Y09_lgb,'Y09_svr':Y09_svr,'Y09_rf':Y09_rf})
tmp.to_csv('matlab/Y09.csv',index=False)
tmp = pd.DataFrame({'Y15_lgb': Y15_lgb,'Y15_svr':Y15_svr,'Y15_rf':Y15_rf})
tmp.to_csv('matlab/Y15.csv',index=False)
tmp = pd.DataFrame({'Y16_lgb': Y16_lgb,'Y16_svr':Y16_svr,'Y16_rf':Y16_rf})
tmp.to_csv('matlab/Y16.csv',index=False)


#%%
train =  = np.vstack([Y09_mean,Y13_mean,Y15_mean,Y16_mean]).T

#%%
# y_pred = Y18_1*0.25 +  Y18_2*0.25 +  Y18_3*0.25 +  Y18_4*0.25
y_pred = Y18_1*0 +  Y18_2*0.3 +  Y18_3*0.7 +  Y18_4*0
# y_pred = Y18_1*0.15 +  Y18_2*0.05 +  Y18_3*0.4 +  Y18_4*0.4

interv = range(8000,8500)
# plt.plot(Y18_1[interv],':')
# plt.plot(Y18_2[interv],':')
# plt.plot(Y18_3[interv],':')
# plt.plot(Y18_4[interv],':')
plt.plot(ref.Y18.values[interv])
plt.plot(ref_ms.Y18.values[interv])
# plt.plot(ref_ms.Y18.values[interv])
plt.plot(y_pred[interv])
# plt.legend(['9','13','15','16','sw','predict'])
plt.legend(['sw','ms','predict'])
mean_squared_error(y_pred, ref_ms.Y18.values)
mean_squared_error(y_pred, ref_prev.Y18.values)
mean_squared_error(y_pred, ref.Y18.values)
mse_AIFrenz(y_pred, ref.Y18.values)

#%%
ref_prev = pd.read_csv('../submit/sample_submission_v37.csv')

ref_ms = pd.read_csv('../submit/submit_8.csv')

#%%
ref_ms.Y18 = y_pred
ref_ms.to_csv('submit/submit_10.csv',index=False)

#%%
ref = pd.read_csv('../submit/sample_submission_v39.csv')
y_pred = Y18_1*0.3 +  Y18_2*0.5 +  Y18_3*0.2

interv = range(3500,4000)
# plt.plot(Y18_1[interv],':')
# plt.plot(Y18_2[interv],':')
# plt.plot(Y18_3[interv],':')
# plt.plot(Y18_4[interv],':')
plt.plot(y_pred[interv])
plt.plot(ref.Y18.values[interv])

#%%

plt.plot(Y13_res_svr[range(100,1000)])
plt.plot(Y13_res_rf[range(100,1000)])

#%%
plt.plot(preds[0][range(2000,3000),1])
plt.plot(preds[1][range(2000,3000),1])
plt.plot(preds[2][range(2000,3000),1])

plt.plot(res_rf[range(500),2])

#%%

Y13_res_rf = np.load('../data_pre/Y13_80day_residual_rf.npy')
Y13_res_lgb = np.load('../data_pre/Y13_80day_residual_lgb.npy')

Y15_res = np.load('../data_pre/Y15_80day_residual_rf.npy')

#%%
for i in range(9):
    interv = range(i*1000,i*1000+500)
    plt.figure()
    plt.plot(Y13_res_rf[interv])
    plt.plot(Y13_res_lgb[interv])

#%%
Y18_1 = Y13_mean + Y13_res
mean_squared_error(Y18_1, ref.Y18.values)

Y18_2 = Y15_mean + Y15_res
mean_squared_error(Y18_2, ref.Y18.values)

Y18_3 = Y16_mean + Y16_res
mean_squared_error(Y18_3, ref.Y18.values)

mean_squared_error(Y18_2, ref.Y18.values)
y_pred = Y18_1 * 0.5 + Y18_2 * 0.5

mean_squared_error(y_pred, ref.Y18.values)
mean_squared_error(y_pred, ref_prev.Y18.values)
mean_squared_error(y_pred, ref_ms.Y18.values)
mean_squared_error(y_pred, ref_ms_prev.Y18.values)

mean_squared_error(ref.Y18.values, ref_ms.Y18.values)

#%%
y_pred = Y18_1*0.6 +  Y18_2*0.4 +  Y18_3*0.0
for i in range(9):
    interv = range(i*1000,i*1000+500)
    plt.figure()
    plt.plot(y_pred[interv])
    plt.plot(ref.Y18.values[interv])
    # plt.plot(ref_ms_prev.Y18.values[interv])
    plt.legend(['pred','sw','ms'])

ref.iloc[:,1] = y_pred
ref.to_csv('submit/submit_12.csv',index=False)
#%%
ref_prev = pd.read_csv('../submit/sample_submission_v37.csv')
ref = pd.read_csv('../submit/sample_submission_v39.csv')
ref_ms = pd.read_csv('../submit/submit_10.csv')
ref_ms_prev = pd.read_csv('../submit/submit_8.csv')

#%%

Y16_svr = np.load('../data_pre/Y16_pred_80day_svr.npy')
Y09_svr = np.load('../data_pre/Y09_pred_80day_svr.npy')
Y15_svr = np.load('../data_pre/Y15_pred_80day_svr.npy')
Y13_svr = np.load('../data_pre/Y13_pred_80day_svr.npy')

# result -- 
Y16_lgb = np.load('../data_pre/Y16_pred_80day_lgb.npy')
Y09_lgb = np.load('../data_pre/Y09_pred_80day_lgb.npy')
Y15_lgb = np.load('../data_pre/Y15_pred_80day_lgb.npy')
Y13_lgb = np.load('../data_pre/Y13_pred_80day_lgb.npy')

# result -- 
Y16_rf = np.load('../data_pre/Y16_pred_80day_rf.npy')
Y09_rf = np.load('../data_pre/Y09_pred_80day_rf.npy')
Y15_rf = np.load('../data_pre/Y15_pred_80day_rf.npy')
Y13_rf = np.load('../data_pre/Y13_pred_80day_rf.npy')
Y18 = np.load('../data_pre/Y18.npy')

Y00_svr = np.load('../data_pre/Y00_pred_80day_svr.npy')
Y00_rf = np.load('../data_pre/Y00_pred_80day_rf.npy')

Y09_mean = np.mean([Y09_lgb,Y09_svr,Y09_rf],axis=0)
Y15_mean = np.mean([Y15_lgb,Y15_svr,Y15_rf],axis=0)
Y16_mean = np.mean([Y16_lgb,Y16_svr,Y16_rf],axis=0)
Y13_mean = np.mean([Y13_lgb,Y13_svr,Y13_rf],axis=0)
Y00_mean = np.mean([Y00_svr,Y00_rf],axis=0)

#%%
Y_18_1 = Y09_mean + preds[4]
Y_18_2 = Y13_mean + preds[0]
Y_18_3 = Y15_mean + preds[1]
Y_18_4 = Y16_mean + preds[2]


#%%
ref_ms = pd.read_csv('../submit/submit_11.csv')
y_pred = Y_18_1 * 0 + Y_18_2 * 0.6 + Y_18_3 * 0.4 + Y_18_4 * 0
mean_squared_error(y_pred,ref_ms.Y18.values)
mean_squared_error(ref_ms.Y18.values,ref.Y18.values)

plt.plot(y_pred[range(1000)])
plt.plot(ref_ms.Y18.values[range(1000)])

#%%

Y00_rf = np.load('../data_pre/Y00_pred_80day_rf.npy')
res = np.load('../data_pre/residual/Y00_80day_residual_rf.npy')
y_pred = Y00_rf + res

ref_ms.Y18 = y_pred
ref_ms.to_csv('submit/submit_14.csv',index=False)

#%%
sensor = 'Y01'
lgb_tmp = np.load('data_pre/'+sensor+'_pred_3day_lgb.npy')
rf_tmp = np.load('data_pre/'+sensor+'_pred_3day_rf.npy')
svr_tmp = np.load('data_pre/'+sensor+'_pred_3day_svr.npy')
Y18 = np.load('../data_pre/Y18.npy')

plt.figure()
plt.plot(lgb_tmp)
plt.plot(rf_tmp)
plt.plot(svr_tmp)
plt.plot(Y18)
plt.legend(['lgb','rf','svr'])

#%%
sensor = 'Y03'
lgb_tmp = np.load('data_pre/'+sensor+'_pred_80day_lgb.npy')
rf_tmp = np.load('data_pre/'+sensor+'_pred_80day_rf.npy')
svr_tmp = np.load('data_pre/'+sensor+'_pred_80day_svr.npy')
ref = pd.read_csv('../submit/sample_submission_v40.csv')
ref = ref.Y18.values
interv = range(3000,4000)
plt.figure()
plt.plot(lgb_tmp[interv])
plt.plot(rf_tmp[interv])
plt.plot(svr_tmp[interv])
# plt.plot(ref[interv])
plt.legend(['lgb','rf','svr'])
# tmp = svr_tmp * 0.1 + rf_tmp * 0.3 + lgb_tmp*0.1
# plt.plot(tmp[interv])

#%%
tmp = np.load('data_pre/Y18_pred_Y00.npy')
interv = range(4000,5000)
plt.plot(tmp[interv])
plt.plot(ref[interv])
ref = pd.read_csv('../submit/sample_submission_v40.csv')
ref = ref.Y18.values
mean_squared_error(tmp, ref)

#%%
Y18_13 = np.load('../data_pre/Y18_pred_Y13.npy')
Y18_00 = np.load('data_pre/Y18_pred_Y00.npy')

#%%
ref = pd.read_csv('../submit/sample_submission_v39.csv')
ref = ref.Y18.values

ref_prev = pd.read_csv('../submit/submit_11.csv')
ref_prev = ref_prev.Y18.values

interv = range(6000,7000)
plt.plot(Y18_00[interv])
plt.plot(Y18_13[interv])
plt.plot(ref[interv])
plt.plot(ref_prev[interv])
mean_squared_error(ref_prev, Y18_13)


#%% 1216,1222
import pandas as pd
tmp = pd.read_csv('../data_pre/train_1.csv')
tmp = pd.read_csv('../data_pre/test.csv')
plt.plot(tmp.loc[:,'X34_diff'])

#%%
plt.plot(tmp.loc[:,'X34_diff'])

#%%
train = pd.read_csv('../data_raw/train.csv')
train = train.iloc[:4752,:]
# train_label = train.loc[:,'label']
train = train.loc[:,'id':'X39']
time = train.id.values % 144
tmp = pd.read_csv('../data_raw/train_X34_diff.csv')
tmp = tmp.iloc[:4752,:]
train['X34_diff'] = tmp.iloc[:,1].values

#%% 
from util import *
train_1, train_label_1 = load_dataset_v2('train1',12, 20, True)
train_2,train_label_2 = load_dataset_v2('train2',12, 20, True)
train = load_dataset_v2('train',12, 20, True)
test = load_dataset_v2('test',12, 20, True)


#%%

plt.plot(train_label_1.loc[:,'Y08':'Y11'].values)
plt.legend(['1','2','3','4'])

#%%
plt.figure()
for label in ['Y01','Y02','Y09','Y15','Y16']:
    data = []
    data.append(np.load('data_pre/'+label+'_pred_3day_svr.npy'))
    data.append(np.load('data_pre/'+label+'_pred_3day_rf.npy'))
    data.append(np.load('data_pre/'+label+'_pred_3day_lgb.npy'))
    data = np.mean(data, axis=0)
    plt.plot(data)
plt.plot(train_label_2.values, linewidth=3)
plt.legend(['Y01','Y02','Y09','Y15','Y16','Y18'])


#%%
trials = load_obj('0402/Y13_lgb') # 0.77
trials = load_obj('0402/Y13_rf') # 0.93
trials = load_obj('0402/Y13_svr') # 0.67

trials = load_obj('0402/Y15_svr') # 0.83
trials = load_obj('0402/Y15_lgb') # 0.89


#%%

tmp1 = np.load('../data_pre/Y13_pred_80day_svr.npy')
tmp2 = np.load('../data_pre/Y13_pred_80day_rf.npy')
tmp3 = np.load('../data_pre/Y13_pred_80day_lgb.npy')

plt.plot(tmp1)
plt.plot(tmp2)
plt.plot(tmp3)

tmp4 = np.load('../predictions/Y13_pred_80day_svr.npy')
tmp5 = np.load('../predictions/Y13_pred_80day_rf.npy')
tmp6 = np.load('../predictions/Y13_pred_80day_lgb.npy')

plt.plot(tmp4)
plt.plot(tmp5)
plt.plot(tmp6)

plt.legend(range(6))

#%%

tmp1 = np.load('../data_pre/Y15_pred_80day_svr.npy')
tmp3 = np.load('../data_pre/Y15_pred_80day_lgb.npy')

plt.plot(tmp1)
plt.plot(tmp3)

tmp4 = np.load('../predictions/Y13_pred_80day_svr.npy')
tmp6 = np.load('../predictions/Y13_pred_80day_lgb.npy')

plt.plot(tmp4)
plt.plot(tmp6)

plt.legend(range(6))

#%%

tmp1 = np.load('../predictions/Y15_pred_80day_svr.npy')
tmp2 = np.load('../predictions/Y15_pred_80day_lgb.npy')
tmp3 = np.load('../data_pre/Y18.npy')

plt.plot(tmp1)
plt.plot(tmp2)
plt.plot(tmp3)

#%%

trials = load_obj('0402/Y15_lgb') # 0.89
trials = load_obj('0402/Y15_svr') # 0.83
trials = load_obj('0402/Y15_rf') # 1.15

trials = load_obj('0404/Y01_lgb') # 0.84
trials = load_obj('0404/Y01_svr') # 0.96
trials = load_obj('0404/Y01_rf') # 1.04

trials = load_obj('0404/Y02_lgb') # 0.92
trials = load_obj('0404/Y02_svr') # 1.05
trials = load_obj('0404/Y02_rf') # 1.08

trials = load_obj('0404/Y16_lgb') # 1.49
trials = load_obj('0404/Y16_svr') 
trials = load_obj('0404/Y16_rf') 

#%%
Y01 = np.load('../predictions/Y01_pred_3day_lgb.npy')
Y02 = np.load('../predictions/Y02_pred_3day_lgb.npy')
Y18 = np.load('../data_pre/Y18.npy')

plt.plot(np.array([Y01, Y02, Y18]).T)

#%%
Y01 = np.load('../predictions/Y01_pred_80day_lgb.npy')
Y02 = np.load('../predictions/Y02_pred_80day_lgb.npy')
plt.plot(np.array([Y01, Y02]).T)
ref = pd.read_csv('../submit/sample_submission_v40.csv')

plt.plot(ref.Y18.values)