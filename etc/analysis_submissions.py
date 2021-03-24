# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 23:42:48 2020

@author: guseh
"""

from util import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ref_1 = pd.read_csv('../submit/sample_submission_v39.csv') # 2.09 0.57

ref_2 = pd.read_csv('../submit/sample_submission_v35.csv') # 2.18 0.64

ref_3 = pd.read_csv('../submit/sample_submission_v33.csv') # 2.27 0.89

ref_4 = pd.read_csv('../submit/submit_8.csv') # 2.48 0.79

ref_5 = pd.read_csv('../submit/submit_10.csv') # 2.61 0.58

ref_6 = pd.read_csv('../submit/submit_11.csv') # x

ref_7 = pd.read_csv('../submit/sample_submission_v26.csv') # 2.5

ref_8 = pd.read_csv('../submit/submit_5.csv') # 2.9

ref_9 = pd.read_csv('../submit/submit_12.csv') # x

mean_squared_error(ref_1.Y18.values, ref_8.Y18.values)
mean_squared_error(ref_1.Y18.values, y_pred)
mse_AIFrenz(ref_1.Y18.values, ref_6.Y18.values)

#%%
for i in range(20):
    interv = range(i*500,i*500+500)
    plt.figure()
    plt.plot(ref_1.Y18.values[interv]) # 2.09
    plt.plot(ref_4.Y18.values[interv],':') # 2.48
    # plt.plot(ref_7.Y18.values[interv]) # 2.5
    plt.plot(ref_6.Y18.values[interv]) # new
    # plt.plot(ref_9.Y18.values[interv],':') # new
    # plt.plot(y_pred[interv]) # new
    plt.legend(['2.09','2.48','pred2'])
    
#%%
y_pred = ref_4.Y18.values*0.6+ ref_9.Y18.values * 0.4
pred_1 = []
pred_2 = []
for i in range(20):
    interv = range(i*500,i*500+500)
    print('1:',mean_squared_error(ref_2.Y18.values[interv], ref_9.Y18.values[interv]))
    pred_1.append(mean_squared_error(ref_2.Y18.values[interv], ref_9.Y18.values[interv]))
    print('2:',mean_squared_error(ref_2.Y18.values[interv], y_pred[interv]))
    pred_2.append(mean_squared_error(ref_2.Y18.values[interv],  y_pred[interv]))

ref.loc[:,1] = y_pred
ref.to_csv('submit/submit_13.csv',index=False)