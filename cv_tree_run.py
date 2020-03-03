# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:40:43 2020

@author: mskim
"""
from cv_tree_objs import *
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
from util import *
import argparse

def parameter_setting(config):
    # LightGBM parameters
    lgb_reg_params = {
        'learning_rate':    hp.choice('learning_rate',    np.arange(0.05, 0.31, 0.05)),
        'max_depth':        hp.choice('max_depth',        np.arange(5, 16, 1, dtype=int)),
        'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
        'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
        'subsample':        hp.uniform('subsample', 0.8, 1),
        'n_estimators':     config['num_boost_round'],
        'seed':             config['seed'],
    }
    lgb_fit_params = { ### Not used
        'eval_metric': 'custom',
        'early_stopping_rounds': 10,
        'verbose': -1
    }
    lgb_para = dict()
    lgb_para['reg_params'] = lgb_reg_params
    lgb_para['fit_params'] = lgb_fit_params
    lgb_para['loss_func' ] = mse_AIFrenz
    return lgb_para

if __name__ == '__main__':
    # configuration
    parser = argparse.ArgumentParser(description='GBDT general fit configuration',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_boost_round", default=100, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--nfold", default=10, type=int)
    parser.add_argument("--max_evals", default=2, type=int)
    parser.add_argument("--filename", default='tmp', type=str)

    args = parser.parse_args()
    config = {}
    config['num_boost_round'] = args.num_boost_round
    config['seed'] = args.seed
    config['nfold'] = args.nfold
    
    # load dataset
    train, _, train_label, _, _, _ = load_dataset('data_raw/')
    train['time'] = train['id'] % 144
    train_label = train_label.drop(columns='id')
    
    bayes_trials_1 = Trials()
    lgb_para = parameter_setting(config)
    obj = HPOpt_cv(train, train_label, config)
    lgb_opt = obj.process(fn_name='lgb_reg', space=lgb_para, trials=bayes_trials_1, algo=tpe.suggest, max_evals=args.max_evals)
    
    # save trial
    save_obj(bayes_trials_1,args.filename)