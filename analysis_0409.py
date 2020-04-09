# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:15:32 2020

@author: guseh
"""


from util import *
import matplotlib.pyplot as plt
import numpy as np

#%% 0409 results
trials = load_obj('0409/Y12_lgb') # 0.59
trials = load_obj('0409/Y13_lgb') # 0.30

#%% residual -- Y12
trials = load_obj('Y12_residual') # 1.64
trials = load_obj('Y12_residual_2') # 1.67
trials = load_obj('Y12_residual_3') # 2.63
trials = load_obj('Y12_residual_4') # 1.69

#%% residual -- Y13
trials = load_obj('Y13_residual') # 2.71
trials = load_obj('Y13_residual_2') # 2.70
trials = load_obj('Y13_residual_3') # 2.70
trials = load_obj('Y13_residual_4') # 2.66

#%% residual -- Y15
trials = load_obj('Y15_residual') # 2.57
