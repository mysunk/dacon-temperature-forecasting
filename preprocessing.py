# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 19:12:37 2020

@author: guseh
"""
import numpy as np
import pandas as pd
from util import *

# load raw data
train_1, train_2, train_label_1, train_label_2, test, sample = load_dataset('data_raw/')

# 하루 단위로 데이터 샘플 만듦
train_1 = make_day_sample(train_1)
train_2 = make_day_sample(train_2)
train_label_1 = make_day_label(train_label_1)
train_label_2 = make_day_label(train_label_2)


