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
        'learning_rate':            hp.uniform('learning_rate',    0.05, 0.3),
        'max_depth':                hp.quniform('max_depth',        10, 200, 1),
        'num_leaves':               hp.quniform('num_leaves',       2, 30, 1),
        'min_data_in_leaf':		hp.quniform('min_data_in_leaf',	10, 300, 10),	
        'reg_alpha':                hp.uniform('reg_alpha',0.0,100.0),
        'reg_lambda':               hp.uniform('reg_lambda',0.0,100.0),
        'min_child_weight':         hp.quniform('min_child_weight', 1, 10, 1),
        'colsample_bytree':         hp.uniform('colsample_bytree', 0.1, 1.0),
        'colsample_bynode':		hp.uniform('colsample_bynode',0.1,1.0),
        'bagging_freq':			hp.quniform('bagging_freq',	1,10,1),
	'tree_learner':			hp.choice('tree_learner',	['serial','feature','data','voting']),
        'subsample':                hp.uniform('subsample', 0.1, 1),
        'boosting':			hp.choice('boosting', ['gbdt','rf','dart']),
        'max_bin':			hp.quniform('max_bin',		10,300,10),
    }
    lgb_reg_params = {
        'learning_rate':            hp.uniform('learning_rate',    0.05, 0.3),
        'max_depth':                hp.quniform('max_depth',        180, 300, 1),
        'num_leaves':               hp.quniform('num_leaves',       2, 20, 1),
        'min_data_in_leaf':		hp.quniform('min_data_in_leaf',	10, 100),	
        'reg_alpha':                hp.uniform('reg_alpha',10.0,20.0),
        'reg_lambda':               hp.uniform('reg_lambda',80.0,100.0),
        'min_child_weight':         hp.quniform('min_child_weight', 1, 10, 1),
        'colsample_bytree':         hp.uniform('colsample_bytree', 0.9, 1.0),
        'colsample_bynode':		hp.uniform('colsample_bynode',0.9,1.0),
        'bagging_freq':			hp.quniform('bagging_freq',	5,10,1),
	'tree_learner':			hp.choice('tree_learner',	['serial','feature','data','voting']),
        'subsample':                hp.uniform('subsample', 0.8, 1),
        'boosting':			hp.choice('boosting', ['gbdt','rf','dart']),
        'max_bin':			hp.quniform('max_bin',		200,300),
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
