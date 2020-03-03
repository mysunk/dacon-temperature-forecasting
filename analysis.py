# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 22:25:48 2020

@author: guseh
"""
from util import *
import matplotlib.pyplot as plt

train_1, train_2, train_label_1, train_label_2, test, sample = load_dataset('data_raw/')

#%% columns
train.columns

#%% 하루 단위로
def plot_feature(data, feature_name, days):
    plt.figure()
    for i in days:
        timestamp = range(i*24*6,(i+1)*24*6)
        plt.plot(timestamp, data[feature_name][list(timestamp)])
    plt.legend(days)
    plt.xlabel('time')
    plt.title(feature_name)
    
def plot_feature_2(data, feature_name, days):
    plt.figure()
    for i in days:
        timestamp = range(i*24*6,(i+1)*24*6)
        plt.plot(range(24*6), data[feature_name][list(timestamp)])
    plt.legend(days)
    plt.xlabel('time')
    plt.title(feature_name)

def plot_features(data, features, days):
    plt.figure()
    timestamp = range(days[0]*24*6,days[-1]*24*6)
    for i in features:
        plt.plot(timestamp, data[i][list(timestamp)])
    plt.legend(features)
    plt.xlabel('time')

#%%
plot_feature(train_label_1, 'Y02',range(2))
plot_feature(train_label_2, 'Y18',range(30,32))

# plot_feature_2(train_1, 'X09',range(10,20))

plot_features(train_1, train_1.columns[12:15], [17, 18])

plot_features(train_label_2, train_label_1.columns[1:10], range(5))



#%% 모든 feature들 n일간
for feature in train_1.columns[1:]:
    plot_feature(train_1,feature ,range(0,30))

#%% 
for feature in train_label_1.columns[1:]:
    plot_feature(train_label_1,feature ,range(2))

#%% Y18 확인

plt.figure()
plt.plot(train_label_2.Y18)

#%% test 확인

for feature in test.columns[1:]:
    plot_feature(test,feature ,range(0,3))
    
#%% For setting
from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'qt')
get_ipython().run_line_magic('matplotlib', 'inline')

#%% Seasonality removal
from statsmodels.tsa.seasonal import seasonal_decompose

seasonal_decompose(train_1, period = 6*24*).seasonal

#%% correlation
train_label_1 = train_label_1.drop(columns='id')
columns = train_label_1.columns
corr = train_label_1[columns].corr()

#%% load pkl result

trials = load_obj('lgb_10fold_3')
