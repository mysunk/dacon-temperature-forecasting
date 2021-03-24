# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:15:32 2020

@author: guseh
"""


from util import *
import matplotlib.pyplot as plt
import numpy as np

#%% Y11 -- 0.297
trials = load_obj('0409/Y11_lgb')
#%% Y12 -- 0.598
trials = load_obj('0409/Y12_lgb')
#%% Y13 -- 0.30
trials = load_obj('0409/Y13_lgb')
#%% Y15 -- 0.286
trials = load_obj('0409/Y15_lgb')
#%% Y16 -- 0.474
trials = load_obj('0409/Y16_lgb')

#%% Y09 -- 0.34
trials = load_obj('0409/Y09_lgb')
#%% Y17 -- 0.67
trials = load_obj('0409/Y17_lgb')
#%%

# day = '3day'
day = '80day'
Y09 = np.ravel(np.load('predictions/Y09_pred_'+ day+'_lgb.npy'))
Y11 = np.ravel(np.load('predictions/Y11_pred_'+ day+'_lgb.npy'))
Y12 = np.ravel(np.load('predictions/Y12_pred_'+ day+'_lgb.npy'))
Y13 = np.ravel(np.load('predictions/Y13_pred_'+ day+'_lgb.npy'))
Y15 = np.ravel(np.load('predictions/Y15_pred_'+ day+'_lgb.npy'))
Y16 = np.ravel(np.load('predictions/Y16_pred_'+ day+'_lgb.npy'))
Y17 = np.ravel(np.load('predictions/Y17_pred_'+ day+'_lgb.npy'))
Y09_residual = np.ravel(np.load('residual_0411/Y09_residual_'+ day+'rf.npy'))
Y11_residual = np.ravel(np.load('residual_0411/Y11_residual_'+ day+'rf.npy'))
Y12_residual = np.ravel(np.load('residual_0411/Y12_residual_'+ day+'rf.npy'))
Y13_residual = np.ravel(np.load('residual_0411/Y13_residual_'+ day+'rf.npy'))
Y15_residual = np.ravel(np.load('residual_0411/Y15_residual_'+ day+'rf.npy'))
Y16_residual = np.ravel(np.load('residual_0411/Y16_residual_'+ day+'rf.npy'))
Y17_residual = np.ravel(np.load('residual_0411/Y17_residual_'+ day+'rf.npy'))

# Y18 = ref_1.Y18.values
Y18 = ref_0.Y18.values
# Y18 = np.load('data_pre/Y18.npy')

#%% Plot
legends = ['Y09','Y11','Y12','Y13','Y15','Y16','Y17']
for i, data in enumerate([Y09,Y11,Y12,Y13,Y15,Y16,Y17]):
    plt.figure()
    plt.plot(data)
    plt.plot(Y18)
    plt.legend([legends[i],'Y18'])

#%% residual

#%% residual -- 2
Y09_residual = np.ravel(np.load('../residual_0411/Y09_residual_3dayrf.npy'))
Y11_residual = np.ravel(np.load('../residual_0411/Y11_residual_3dayrf.npy'))
Y12_residual = np.ravel(np.load('../residual_0411/Y12_residual_3dayrf.npy'))
Y13_residual = np.ravel(np.load('../residual_0411/Y13_residual_3dayrf.npy'))
Y15_residual = np.ravel(np.load('../residual_0411/Y15_residual_3dayrf.npy'))
Y16_residual = np.ravel(np.load('../residual_0411/Y16_residual_3dayrf.npy'))
Y17_residual = np.ravel(np.load('../residual_0411/Y17_residual_3dayrf.npy'))

#%% residual만 plot
legends = ['Y09','Y11','Y12','Y13','Y15','Y16','Y17']
for i, data in enumerate([Y09,Y11,Y12,Y13,Y15,Y16,Y17]):
    residual = eval(legends[i]+'_residual')
    plt.figure()
    plt.plot(residual)
    plt.legend([legends[i]])

#%% residual까지 plot
legends = ['Y09','Y11','Y12','Y13','Y15','Y16','Y17']
for i, data in enumerate([Y09,Y11,Y12,Y13,Y15,Y16,Y17]):
    residual = eval(legends[i]+'_residual')
    plt.figure()
    plt.plot(data+residual)
    plt.plot(Y18)
    plt.legend([legends[i],'Y18'])

#%% 서버에서 얻은 residual
trials = load_obj('0411/Y09_residual_rf') # 2.56
trials = load_obj('0411/Y11_residual_rf') # 2.85
trials = load_obj('0410/Y12_residual_rf') # 1.59
trials = load_obj('0410/Y13_residual_rf') # 2.55
trials = load_obj('0410/Y15_residual_rf') # 2.31
trials = load_obj('0411/Y16_residual_rf') # 2.00
trials = load_obj('0411/Y17_residual_rf') # 3.17

#%% mse 비교
for i, data in enumerate([Y09,Y11,Y12,Y13,Y15,Y16,Y17]):
    residual = eval(legends[i]+'_residual')
    data = data + residual
    print(mean_squared_error(data, Y18))
#%% 80일치 mse 비교
legends = ['Y09','Y11','Y12','Y13','Y15','Y16','Y17']
for i, data in enumerate([Y09,Y11,Y12,Y13,Y15,Y16,Y17]):
    residual = eval(legends[i]+'_residual')
    data = data + residual
    print(mean_squared_error(data, Y18))

#%% mse 비교
mean_squared_error(Y18_Y11, Y18)
mean_squared_error(Y18_Y12, Y18)
mean_squared_error(Y18_Y13, Y18)
mean_squared_error(Y18_Y15, Y18)
mean_squared_error(Y18_Y16, Y18)

y_pred = Y18_Y11 * 0.0 + Y18_Y12 * 0.3 + Y18_Y13 * 0.2 + Y18_Y15 * 0.2 + Y18_Y16 * 0.3
mean_squared_error(y_pred, Y18)



#%%
for i in range(11):
    for j in range(11):
        rest = 1 - 0.1 * i
        y_pred = Y18_Y12 * 0.1 * i + Y18_Y13 * rest * 0.1 * j + Y18_Y15 * rest * (1-0.1 * j)
        print(f'for {i} and {j}, {mse_AIFrenz(y_pred, Y18)}')
        

#%%
Y11 = np.ravel(np.load('../predictions/Y11_pred_80day_lgb.npy'))
Y12 = np.ravel(np.load('../predictions/Y12_pred_80day_lgb.npy'))
Y13 = np.ravel(np.load('../predictions/Y13_pred_80day_lgb.npy'))
Y15 = np.ravel(np.load('../predictions/Y15_pred_80day_lgb.npy'))
Y16 = np.ravel(np.load('../predictions/Y16_pred_80day_lgb.npy'))

Y18_Y11 = Y11 + np.load('../residual_0410/Y11_residual_80dayrf.npy')
Y18_Y12 = Y12 + np.load('../residual_0410/Y12_residual_80dayrf.npy')
Y18_Y13 = Y13 + np.load('../residual_0410/Y13_residual_80dayrf.npy')
Y18_Y15 = Y15 + np.load('../residual_0410/Y15_residual_80dayrf.npy')
Y18_Y16 = Y16 + np.load('../residual_0410/Y16_residual_80dayrf.npy')

plt.plot(np.array([Y18_Y12,Y18_Y13,Y18_Y15,Y18_Y16]).T)
plt.plot(ref_1.Y18.values)
plt.plot(ref_2.Y18.values)
plt.legend(['Y12','Y13','Y15','Y16','ref1','ref2'])

#%%
ref_0 = pd.read_csv('submit/sample_submission_v39.csv') # 2.08
ref_1 = pd.read_csv('submit/submit_25.csv') # 1.71
ref_2 = pd.read_csv('submit/submit_27.csv') # 2.06
ref_3 = pd.read_csv('submit/sample_submission_v40.csv') # 1.8
ref_4 = pd.read_csv('submit/submit_11.csv') # 1.97
ref_5 = pd.read_csv('submit/submit_28.csv') # 1.75
ref_6 = pd.read_csv('submit/submit_24.csv') # 1.72
ref_7 = pd.read_csv('submit/submit_29.csv') # 1.99
ref_8 = pd.read_csv('submit/submit_30.csv') # 1.78

data = Y18_Y16
mean_squared_error(ref_2.Y18.values, ref_6.Y18.values)
mean_squared_error(ref_2.Y18.values, data)

#%%
y_pred = Y18_Y12 * 0.2 + Y18_Y13 * 0.5+ Y18_Y15 * 0.3+ Y18_Y16 * 0
# y_pred = ref_2.Y18.values * 0.8 + Y18_Y16 * 0
ensemble = ref_0.Y18.values * 0.5 + y_pred * 0.5
ensemble2 = ref_0.Y18.values * 0.4 + ref_2.Y18.values * 0.6

#%%

y_pred = Y18_Y12 * 0.5 + Y18_Y13 * 0.3 + Y18_Y15 * 0.2+ Y18_Y16 * 0
ensemble = ref_0.Y18.values * 0.4 + y_pred * 0.6

result = []
for i in [ref_0,ref_1, ref_2,ref_3,ref_4,ref_5,ref_6, ref_7]:
    result.append(mean_squared_error(i.Y18.values, y_pred))

result2 = []
for i in [ref_0,ref_1, ref_2,ref_3,ref_4,ref_5,ref_6, ref_7]:
    result2.append(mean_squared_error(i.Y18.values, ensemble))
    

#%%
y_pred = Y18_Y12 * 0.5 + Y18_Y13 * 0.3 + Y18_Y15 * 0.2+ Y18_Y16 * 0
ensemble = ref_0.Y18.values * 0.4 + y_pred * 0.6
print(mean_squared_error(ensemble, ref_1.Y18.values))

#%%
plt.plot(ref_2.Y18.values)
plt.plot(ref_6.Y18.values)
plt.plot(y_pred)
plt.plot(ensemble)

#%%
mean_squared_error(ref_8.Y18.values,ref_2.Y18.values)

#%%

y_pred = (Y09 + Y09_residual) * 0.05 + (Y11 + Y11_residual) * 0.05 + (Y12 + Y12_residual) * 0.35 +\
    (Y13 + Y13_residual) * 0.2 + (Y15 + Y15_residual) * 0.2+ (Y16 + Y16_residual) * 0.1 +(Y17 + Y17_residual) * 0.05
    
result = []
for i in [ref_0,ref_1, ref_2,ref_3,ref_4,ref_5,ref_6, ref_7, ref_8]:
    result.append(mean_squared_error(i.Y18.values, y_pred))

ensemble = ref_0.Y18.values * 0.3 + y_pred * 0.7

result2 = []
for i in [ref_0,ref_1, ref_2,ref_3,ref_4,ref_5,ref_6, ref_7, ref_8]:
    result2.append(mean_squared_error(i.Y18.values, ensemble))
    
#%%
# plt.plot(ref_0.Y18.values)
plt.plot(ref_1.Y18.values)
plt.plot(ref_2.Y18.values)
plt.plot(y_pred)
plt.plot(ensemble)