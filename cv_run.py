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
parser.add_argument("--nfolds", default=432, type=int)
parser.add_argument("--filename", default='rf_144fold_1', type=str)
parser.add_argument("--clf_type", default='rf', type=str)
args = parser.parse_args()

# load raw data

train = np.load('data_npy/train_pred_2.npy')
train = train[:,0].reshape(-1,1)
_, _, _, train_label, _, _ = load_dataset('data_raw/')
# train_1['time'] = train_1['id'] % 144
# train_2['time'] = train_2['id'] % 144
# train_label_1 = train_label_1.drop(columns='id')
# train_1 = train_1.drop(columns='id')
train_label = train_label.drop(columns='id')

# training phase
bayes_trials_1 = Trials()
obj = HPOpt(args.random_state, args.clf_type)
opt = obj.process(fn_name=args.clf_type+'_cv',train_set = (train, train_label.values),nfolds=args.nfolds,
                trials=bayes_trials_1, algo=tpe.suggest, max_evals=args.max_evals)

# save trials
save_obj(bayes_trials_1,args.filename)
