"""
Created on Mon Mar  2 22:44:55 2020

@author: guseh
"""
# packages
import argparse
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
from util import *
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, STATUS_FAIL
from functools import partial
from sklearn.multioutput import MultiOutputRegressor
# models
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

class Tuning_model(object):
    
    def __init__(self):
        self.random_state = 0
        self.space = {}
    
    # parameter setting
    def eln_space(self):
        self.space =  {
            'max_iter':                 hp.quniform('max_iter',100, 1000,1),
             "alpha":                   hp.loguniform('alpha',np.log(0.0001),np.log(1000)),
             'l1_ratio':                hp.uniform('l1_ratio',0.0, 1.0),
             'random_state' :           self.random_state,
             }
    
    def rf_space(self):
        self.space =  {
            'max_depth':                hp.quniform('max_depth',2, 30,1),
            'min_samples_leaf':         hp.quniform('min_samples_leaf', 1,50,1),
            'max_features':             hp.quniform('max_features', 2,39,1),
            'min_samples_split':        hp.quniform('min_samples_split', 1,10,1),
            'n_estimators':             hp.quniform('n_estimators', 100,1000,1),
            #'criterion':               hp.choice('criterion', ["gini", "entropy"]),
            'random_state' :            self.random_state,
           }
    
    def svr_space(self):
        self.space = {
            'kernel':                   hp.choice('kernel',['linear', 'rbf','poly']),
            'C':                        hp.uniform('C',1,10),
            'gamma':                    hp.loguniform('gamma',np.log(1e-7),np.log(1e-1)),
            'epsilon':                  hp.uniform('epsilon',0.1,0.9),
            }
        
    def lgb_space(self):
        # LightGBM parameters
        self.space = {
            'learning_rate':            hp.uniform('learning_rate',    0.001, 0.2),
            'max_depth':                -1,
            'num_leaves':               hp.quniform('num_leaves',       5, 20, 1), 
            'min_data_in_leaf':		    hp.quniform('min_data_in_leaf',	50, 300, 1),	# overfitting 안되려면 높은 값
            'reg_alpha':                hp.uniform('reg_alpha',0.1,0.95),
            'reg_lambda':               hp.uniform('reg_lambda',0.1, 0.95),
            'min_child_weight':         hp.quniform('min_child_weight', 1, 30, 1),
            'colsample_bytree':         hp.uniform('colsample_bytree', 0.01, 1.0),
            'colsample_bynode':		    hp.uniform('colsample_bynode',0.01,1.0),
            'bagging_freq':			    hp.quniform('bagging_freq',	1,20,1),
            'tree_learner':			    hp.choice('tree_learner',	['serial','feature','data','voting']),
            'subsample':                hp.uniform('subsample', 0.01, 1.0),
            'boosting':			        hp.choice('boosting', ['gbdt','rf']),
            'max_bin':			        hp.quniform('max_bin',		3,50,1), # overfitting 안되려면 낮은 값
            "min_sum_hessian_in_leaf": hp.quniform('min_sum_hessian_in_leaf',       5, 15, 1), 
            'random_state':             self.random_state,
            'n_jobs':                   -1,
            'metrics':                  'None'
        }
            
    # optimize
    def process(self, clf_name, train_set, nfolds, trials, algo, max_evals):
        fn = getattr(self, clf_name+'_cv')
        space = getattr(self, clf_name+'_space')
        space()
        fmin_objective = partial(fn, train_set=train_set,nfolds=nfolds)
        try:
            result = fmin(fn=fmin_objective, space=self.space, algo=algo, max_evals=max_evals, trials=trials)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result, trials
    
    # objective function
    def eln_cv(self, params, train_set, nfolds):
        params = make_param_int(params, ['max_iter'])
        model = ElasticNet(**params)
        score= make_scorer(mse_AIFrenz, greater_is_better=True)
        cv_results = cross_val_score(model, train_set[0], train_set[1], cv=nfolds,n_jobs=-1, verbose=0, scoring=score)
        # print(cv_results)
        cv_loss = np.mean(cv_results)
        print('k-fold loss is',cv_results)
        # Dictionary with information for evaluation
        return {'loss': cv_loss, 'params': params, 'status': STATUS_OK}
    
    def rf_cv(self, params, train_set, nfolds):
        params = make_param_int(params,['max_depth','max_features','n_estimators','min_samples_split','min_samples_leaf'])
        params = make_param_int(params,['n_estimators','max_features','max_depth'])
        model = RandomForestRegressor(**params)
        # score= make_scorer(mean_squared_error, greater_is_better=True)
        score= make_scorer(mse_AIFrenz, greater_is_better=True)
        cv_results = cross_val_score(model, train_set[0], train_set[1], cv=nfolds,n_jobs=-1, verbose=0, scoring=score)
        cv_loss = np.mean(cv_results)
        print('k-fold loss is',cv_results)
        # Dictionary with information for evaluation
        return {'loss': cv_loss, 'params': params, 'status': STATUS_OK}
    
    def svr_cv(self, params, train_set, nfolds):
        model =SVR(**params)
        score= make_scorer(mse_AIFrenz, greater_is_better=True)
        cv_results = cross_val_score(model, train_set[0], train_set[1], cv=nfolds,n_jobs=-1, verbose=0, scoring=score)
        cv_loss = np.mean(cv_results)
        print('k-fold loss is',cv_results)
        # Dictionary with information for evaluation
        return {'loss': cv_loss, 'params': params, 'status': STATUS_OK}
    """
    def lgb_cv(self, params, train_set, nfolds):
        params = make_param_int(params, ['max_depth','num_leaves','min_data_in_leaf',
                                     'min_child_weight','bagging_freq','max_bin','min_sum_hessian_in_leaf'])
        # cv_results = lgb.cv(params, train_set, num_boost_round=1000,nfold=nfolds,stratified=True,verbose_eval=True,
        #                     feval=mse_AIFrenz_lgb, early_stopping_rounds=10)
        dtrain = lgb.Dataset(train_set[0], label = train_set[1])
        cv_results = lgb.cv(params, dtrain, num_boost_round=100,nfold=nfolds,stratified=False,verbose_eval=True,
                             metrics="l2", early_stopping_rounds=10)
        best_loss = min(cv_results['l2-mean'])
        # Dictionary with information for evaluation
        return {'loss': best_loss, 'params': params, 'status': STATUS_OK}
    """
    def lgb_cv(self, params, train_set, nfolds): # 실제 학습이랑 성능 차이가 너무 심해서..
        params = make_param_int(params, ['max_depth','num_leaves','min_data_in_leaf',
                                     'min_child_weight','bagging_freq','max_bin','min_sum_hessian_in_leaf'])
        train = train_set[0]
        traon_label = train_set[1]
        losses = []
        kf = KFold(n_splits=nfolds,random_state=None, shuffle=False)
        for i, (train_index, test_index) in enumerate(kf.split(train, train_label)):
            if isinstance(train, (np.ndarray, np.generic) ): # if numpy array
                x_train = train[train_index]
                y_train = train_label[train_index]
                x_test = train[test_index]
                y_test = train_label[test_index]
            else: # if dataframe
                x_train = train.iloc[train_index]
                y_train = train_label.iloc[train_index]
                x_test = train.iloc[test_index]
                y_test = train_label.iloc[test_index]
            dtrain = lgb.Dataset(x_train, label=y_train)
            dvalid = lgb.Dataset(x_test, label=y_test)
            model = lgb.train(params, train_set = dtrain,  
                              valid_sets=[dtrain, dvalid],num_boost_round=1000,verbose_eval=False,feval=mse_AIFrenz_lgb,
                                     early_stopping_rounds=10)
            losses.append(model.best_score['valid_1']['mse_modified'])
        return {'loss': np.mean(losses,axis=0),'params':params ,'status': STATUS_OK}
    
if __name__ == '__main__':
    
    # load config
    parser = argparse.ArgumentParser(description='Dacon temperature regression',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--method', default='rf', choices=['lgb', 'eln', 'rf','svr'])
    parser.add_argument('--max_evals', default=300,type=int)
    parser.add_argument('--save_file', default='Y16_residual_2')
    parser.add_argument('--nfold', default=10,type=int)
    parser.add_argument('--label', default='Y16')
    args = parser.parse_args()
    
    label = args.label
    train, train_label = load_dataset_v2('train2',12, 20, True)
    # del train['time']
    
    data = []
    # data.append(np.load('predictions/'+label+'_pred_3day_svr.npy'))
    # data.append(np.load('predictions/'+label+'_pred_3day_rf.npy'))
    data.append(np.load('predictions/'+label+'_pred_3day_lgb.npy'))
    
    data = np.mean(data, axis=0)
    train_label = np.ravel(train_label.values - data)
    train = train.values

    # main
    clf = args.method
    bayes_trials = Trials()
    obj = Tuning_model()
    tuning_algo = tpe.suggest
    # tuning_algo = tpe.rand.suggest # -- random search
    obj.process(args.method, ((train, train_label)), args.nfold, 
                           bayes_trials, tuning_algo, args.max_evals)
    
    # save trial
    save_obj(bayes_trials.results,args.save_file)