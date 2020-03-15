# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 23:25:58 2020

@author: guseh
"""
from cv_obj import *
from hyperopt import tpe, fmin, Trials
import argparse
from util import *

# argparse
parser = argparse.ArgumentParser(description='Fit configuration',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--max_evals", default=1000, type=int)
parser.add_argument("--random_state", default=0, type=int)
parser.add_argument("--nfolds", default=10, type=int)
parser.add_argument("--filename", default='tmp_eln', type=str)
parser.add_argument("--clf_type", default='eln', type=str)
args = parser.parse_args()

# load dataset
train = pd.read_csv('data_raw/train.csv')
test = pd.read_csv('data_raw/test.csv')

# split data and label
train = train.loc[:,'id':'X39']
train['time'] = train.id.values % 144
train = train.drop(columns = 'id')
train_label = pd.read_csv('data_npy/Y_18.csv')

# declare dataset
train = add_profile(train, ['X00']) # 기온만 추가
# train = train.iloc[144:,:]
train_label = train_label[144:]

# training phase
bayes_trials_1 = Trials()
obj = HPOpt(args.random_state, args.clf_type)
opt = obj.process(fn_name=args.clf_type+'_cv',train_set = (train.values, train_label.values),nfolds=args.nfolds,
                trials=bayes_trials_1, algo=tpe.suggest, max_evals=args.max_evals)

# save trials
save_obj(bayes_trials_1,args.filename)