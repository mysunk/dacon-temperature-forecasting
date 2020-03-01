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

plot_feature(train_label_2, 'Y18',range(3))

plt.

# plot_feature_2(train_1, 'X09',range(10,20))

plot_features(train_1, train_1.columns[12:15], [17, 18])

plot_features(train_label_1, train_label_1.columns[1:10], range(5))

#%% Y 보기
for i in range(24*6*30,24*6*33):
    print(i,':',train.Y18[i])

    
#%% 모든 feature들 n일간
for feature in train_1.columns[1:]:
    plot_feature(train_1,feature ,range(0,10))

#%% 
for feature in train_label_1.columns[1:]:
    plot_feature(train_label_1,feature ,range(10))

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
    