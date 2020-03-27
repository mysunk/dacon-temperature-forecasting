# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:19:59 2020

@author: guseh
"""
from util import *
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, mean_squared_error

# clf
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

def lgb_param():
    lgb_param = {
        'bagging_freq' :            2,
        'boosting' :                'gbdt',
        'colsample_bynode' :        0.5614113707540148,
        'colsample_bytree' :        0.5236228311328034,
        'learning_rate' :           0.11074930666097238,
       ' max_bin' :                 100,
        'max_depth' :               10,
        'min_child_weight' :        4,
        'min_data_in_leaf' :        30,
        'num_leaves' :              20,
        'reg_alpha' :               0.0346297125460458,
        'reg_lambda':               3.641221146386232,
        'subsample':                0.891312377550311,
        'tree_learner':             'feature',
        'random_state':             0,
        'n_jobs':                   -1,
    }
    return lgb_param

def eln_param():
    eln_param =  {
        'max_iter':                 10,
         "alpha":                   1,
         'l1_ratio':                0.1,
         'random_state' :           0,
         }

def rf_param(self):
    rf_param =  {
        'max_depth':                10,
        'max_features':             10,
        'n_estimators':             100,
        #'criterion':               hp.choice('criterion', ["gini", "entropy"]),
        'random_state' :            0,
       }

def svr_param(self):
    svr_param = {
        'kernel':                   'linear',
        'C':                        1,
        'gamma':                    1e-3,
        'epsilon':                  0.2,
        'random_state' :           0,
        }

if __name__ == '__main__':
    # load cv result
    # param = lgb_param() # pre-defined param
    
    # param_sequence = [33, 65, 66]
    param_sequence = [70, 69, 43]
    preds = []
    loss_results = []
    for tries in param_sequence:
        trials = load_obj('tmp')
        param = trials[tries]['params']
        param['metric']='l2'
        
        # User
        N_T = 12
        N_S = 20
        nfold = 30
        
        #============================================= load & pre-processing ==================================================
        # split data and label
        train_1, train_2, train_label_1, train_label_2, test, sample = load_dataset('data_raw/')
        train = pd.read_csv('data_raw/train.csv')
        train_label = train_label_1.loc[:,'Y15']
        
        # if for test sample
        train = pd.concat([train,test],axis=0).reset_index(drop=True)
        
        # add and delete feature
        train = train.loc[:,'id':'X39']
        drop_feature = ['id','X14','X16','X19']
        time = train.id.values % 144
        train = train.drop(columns = drop_feature)
        train['X11_diff'] = irradiance_difference(train.X11.values) # 누적을 difference로 바꿈
        train['X34_diff'] = irradiance_difference(train.X34.values)
        train['time'] = time
        train = train.loc[:,['time','X00','X07','X30','X31','X34','X34_diff']]
        train = add_profile_v4(train, 'X31',N_T) # 온도
        train = add_profile_v4(train, 'X34_diff',N_S) # 일사량
        
        # test = train.iloc[4320:,:]
        # train = train.iloc[:4320,:]
        
        test = train.iloc[4752:,:]
        train = train.iloc[:4320,:]
        #============================================= load & pre-processing ==================================================
        
        if nfold==0:
            dtrain = lgb.Dataset(train, label=train_label)
            model = lgb.train(param, train_set = dtrain,valid_sets = [dtrain], num_boost_round=1000,verbose_eval=True,
                                         early_stopping_rounds=10)
            y_pred = model.predict(test)
        else:
            losses = np.zeros((nfold,2)) # 0:train, 1:val
            preds_test = []
            models = []
            kf = KFold(n_splits=nfold, random_state=None, shuffle=False)
            for i, (train_index, test_index) in enumerate(kf.split(train, train_label)):
                print(i,'th fold training')
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
                model = lgb.train(param, train_set = dtrain,valid_sets = [dtrain, dvalid], num_boost_round=1000,verbose_eval=True,
                                         early_stopping_rounds=10)
                models.append(model)
                preds_test.append(model.predict(test))
                losses[i,0] = model.best_score['training']['l2']
                losses[i,1] = model.best_score['valid_1']['l2']
            y_pred = np.mean(preds_test, axis=0)
        loss_results.append(losses)
        preds.append(y_pred)
    
    y_pred = np.mean(preds,axis=0)
    np.save('data_npy/Y15_pred_80day.npy',y_pred)
    
    # check performance
    # ref = pd.read_csv('submit/sample_submission_v26.csv')
    # print('diff with sw is',mean_squared_error(ref.Y18.values, y_pred))
    
    """ other classifier
    # rf
    model = RandomForestRegressor(**param)
    model.fit(train, train_label)
    """