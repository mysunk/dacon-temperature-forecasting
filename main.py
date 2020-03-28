# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:19:59 2020

@author: guseh
"""
from util import *
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.preprocessing import StandardScaler

# clf
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.svm import SVR

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
    
    # user
    param_sequence = [32] 
    sensor = 'Y16'
    trials = load_obj('0327/'+sensor)
    method = 'svr'
    test_type = '3day' # '80day'
    save_name = 'data_pre/'+sensor+'_pred_'+test_type+'_'+method+'.npy'
    
    preds = []
    loss_results = []
    
    for tries in param_sequence:
        
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
        train_label = train_label_1.loc[:,sensor]
        
        if test_type == '80day':
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
        
        if test_type == '3day':
            test = train.iloc[4320:,:]
            train = train.iloc[:4320,:]
        elif test_type == '80day':
            test = train.iloc[4752:,:]
            train = train.iloc[:4320,:]
        
        # standart scaler
        scaler = StandardScaler()
        train.loc[:,:] = scaler.fit_transform(train.values)
        test.loc[:,:] = scaler.transform(test.values)
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
                # train test split
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
                
                # w.r.t method
                if method == 'lgb':
                    dtrain = lgb.Dataset(x_train, label=y_train)
                    dvalid = lgb.Dataset(x_test, label=y_test)
                    model = lgb.train(param, train_set = dtrain,valid_sets = [dtrain, dvalid], num_boost_round=1000,verbose_eval=True,
                                             early_stopping_rounds=10)
                elif method == 'svr':
                    if not i: del param['metric']
                    model = SVR(**param)
                    model.fit(x_train, y_train)
                elif method == 'rf':
                    if not i: del param['metric']
                    model = RandomForestRegressor(**param)
                    model.fit(x_train, y_train)
                # for evaluation
                train_pred = model.predict(x_train)
                valid_pred = model.predict(x_test)
                losses[i,0]= mean_squared_error(y_train, train_pred)
                losses[i,1]= mean_squared_error(y_test, valid_pred)
                # test
                preds_test.append(model.predict(test))
            # average k-fold results
            y_pred = np.mean(preds_test, axis=0)
        loss_results.append(losses)
        preds.append(y_pred)
    y_pred = np.mean(preds,axis=0)
    np.save(save_name,y_pred)
    
    
    """ other classifier
    # rf
    
    """