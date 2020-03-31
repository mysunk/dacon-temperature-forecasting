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
    return eln_param

def rf_param():
    rf_param =  {
        'max_depth':                10,
        'max_features':             10,
        'n_estimators':             100,
        #'criterion':               hp.choice('criterion', ["gini", "entropy"]),
        'random_state' :            0,
       }
    return rf_param

def svr_param():
    svr_param = {
        'kernel':                   'rbf',
        'C':                        5.6,
        'gamma':                    0.035,
        'epsilon':                  0.23,
        'random_state' :           0,
        }
    return svr_param

if __name__ == '__main__':
    # load cv result
    # user
    # trials = load_obj('0328/'+sensor)
    sensor = 'Y13'
    trials = load_obj('0329/Y13_svr')
    param_num = 30
    random_seeds = [0]
    method = 'svr'
    
    test1_savename = 'data_pre/'+sensor+'_pred_3day_'+method+'.npy'
    test2_savename = 'data_pre/'+sensor+'_pred_80day_'+method+'.npy'
    
    preds_all_test1 = []
    preds_all_test2 = []
    loss_results = []
    
    for seeds in random_seeds:
        
        param = trials[param_num]['params']
        param['metric']='l2'
        # param['random_state'] = seeds
        
        # User
        N_T = 12
        N_S = 20
        nfold = 30
        
        #============================================= load & pre-processing ==================================================
        # split data and label
        train_1, train_2, train_label_1, train_label_2, test, sample = load_dataset('data_raw/')
        train = pd.read_csv('data_raw/train.csv')
        train_label = train_label_1.loc[:,sensor]
        
        
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
        
        test1 = train.iloc[4320:4752,:]
        test2 = train.iloc[4752:,:]
        train = train.iloc[:4320,:]

    
        # transform
        scaler = StandardScaler()
        train.loc[:,:] = scaler.fit_transform(train.values)
        test1.loc[:,:] = scaler.transform(test1.values)
        test2.loc[:,:] = scaler.transform(test2.values)
        
        
        #============================================= load & pre-processing ==================================================
        
        if nfold==0:
            dtrain = lgb.Dataset(train, label=train_label)
            model = lgb.train(param, train_set = dtrain,valid_sets = [dtrain], num_boost_round=1000,verbose_eval=True,
                                         early_stopping_rounds=10)
            y_pred = model.predict(test)
        else:
            losses = np.zeros((nfold,2)) # 0:train, 1:val
            preds_test1 = []
            preds_test2 = []
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
                    model = RandomForestRegressor(n_jobs=-1,**param)
                    model.fit(x_train, y_train)
                # for evaluation
                train_pred = model.predict(x_train)
                valid_pred = model.predict(x_test)
                losses[i,0]= mean_squared_error(y_train, train_pred)
                losses[i,1]= mean_squared_error(y_test, valid_pred)
                # test
                preds_test1.append(model.predict(test1))
                preds_test2.append(model.predict(test2))
            # average k-fold results
            y_pred_test1 = np.mean(preds_test1, axis=0)
            y_pred_test2 = np.mean(preds_test2, axis=0)
        loss_results.append(losses)
        preds_all_test1.append(y_pred_test1)
        preds_all_test2.append(y_pred_test2)
    y_pred_test1 = np.mean(preds_all_test1,axis=0)
    y_pred_test2 = np.mean(preds_all_test2,axis=0)
    np.save(test1_savename,y_pred_test1)
    np.save(test2_savename,y_pred_test2)
    
