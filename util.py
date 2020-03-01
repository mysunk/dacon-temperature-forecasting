# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 21:24:21 2020

@author: guseh
"""

import pandas as pd

def load_dataset(path):
    train = pd.read_csv(path+'train.csv')
    test = pd.read_csv(path+'test.csv')
    sample = pd.read_csv(path+'sample_submission.csv')
    
    # split data and label
    train_1 = train.iloc[:24*6*30,:41] # 30일간 X00 ~ X39
    train_2 = train.iloc[24*6*30:,:41] # 3일간 X00 ~ X39
    train_label_1 = train.iloc[:24*6*30,41:-1] # 30일간 Y00 ~ Y17
    train_label_1.insert(0,'id',train.id[:24*6*30])
    train_label_2 = train.iloc[24*6*30:,-1] # 3일간 Y18
    train_label_2 = pd.concat( [train.id[24*6*30:],train_label_2], axis=1)
    
    return train_1, train_2, train_label_1, train_label_2, test, sample