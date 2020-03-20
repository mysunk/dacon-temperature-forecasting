# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:40:43 2020

@author: mskim
"""
from optimization import *
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials, hp
from util import *
import argparse
from regression import *

def parsing(file_name, method):
    config = ConfigParser()
    config.read(file_name)
    # load data params
    data_params = {}
    data_params['train_dataset'] = config.get('data', 'train_dataset')
    data_params['target_dataset'] = config.get('data', 'target_dataset')
    data_params['analysis_type'] = config.get('data', 'analysis_type') # tuning or training
    data_params['filename'] = config.get('data', 'filename')
    data_params['max_evals'] = config.getint('data', 'max_evals')
    data_params['nfold'] = config.getint('data', 'nfold')
    data_params['N'] = config.getint('data', 'N')
    
    # load clf params
    reg_params = {}
    fit_params = {}
    if method == 'lgb':
        fit_params['num_boost_round'] = config.getint(method, 'num_boost_round')
        fit_params['early_stopping_rounds'] = config.getint(method, 'early_stopping_rounds')
        fit_params['verbose_eval'] = config.getint(method, 'verbose_eval')
        fit_params['seed'] = config.getint(method, 'seed')
        if data_params['analysis_type'] == 'training': # training
            reg_params['bagging_freq'] = config.getint(method+'_param', 'bagging_freq')
            reg_params['boosting'] = config.get(method+'_param', 'boosting')
            reg_params['colsample_bynode'] = config.getfloat(method+'_param', 'colsample_bynode')
            reg_params['colsample_bytree'] = config.getfloat(method+'_param', 'colsample_bytree')
            reg_params['learning_rate'] = config.getfloat(method+'_param', 'learning_rate')
            reg_params['max_bin'] = config.getint(method+'_param', 'max_bin')
            reg_params['max_depth'] = config.getint(method+'_param', 'max_depth')
            reg_params['min_child_weight'] = config.getint(method+'_param', 'min_child_weight')
            reg_params['min_data_in_leaf'] = config.getint(method+'_param', 'min_data_in_leaf')
            reg_params['num_leaves'] = config.getint(method+'_param', 'num_leaves')
            reg_params['reg_alpha'] = config.getfloat(method+'_param', 'reg_alpha')
            reg_params['reg_lambda'] = config.getfloat(method+'_param', 'reg_lambda')
            reg_params['subsample'] = config.getfloat(method+'_param', 'subsample')
            reg_params['tree_learner'] = config.get(method+'_param', 'tree_learner')
    model_params = {'reg_params':reg_params, 'fit_params':fit_params}
    return data_params, model_params

if __name__ == '__main__':
    
    # load config
    parser = argparse.ArgumentParser(description='Dacon temperature regression',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ini_file', default='training_v1')
    parser.add_argument('--method', default='lgb', choices=['lgb', 'xgb', 'ctb'])
    args = parser.parse_args()
    data_params, model_params = parsing('./config/'+args.ini_file+'.ini', args.method)
    
    # load dataset
    if data_params['train_dataset'][-3:] == 'csv': train = pd.read_csv(data_params['train_dataset'])
    else:train = np.load(data_params['train_dataset'])
    if data_params['target_dataset'][-3:] == 'csv': train_label = pd.read_csv(data_params['target_dataset'])
    else: train_label = np.load(data_params['target_dataset'])
    
    # load dataset -- 다시
    train = pd.read_csv('data_raw/train.csv')
    test = pd.read_csv('data_raw/test.csv')

    # split data and label
    train = train.loc[:,'id':'X39']
    train['time'] = train.id.values % 144
    train = train.drop(columns = 'id')
    train_label = pd.read_csv('data_npy/Y_18.csv')
    
    # declare dataset
    N = data_params['N']
    train = add_profile_v2(train, ['X00'],N) # 기온만 추가
    # train = train.iloc[144:,:]
    train_label = train_label[N:]
    
    # main
    if data_params['analysis_type'] == 'tuning':
        bayes_trials = Trials()
        space = lgb_space(model_params['fit_params'])
        obj = HyperOptimize(train, train_label, data_params['nfold'])
        tuning_algo = tpe.suggest  # tpe.rand.sugggest -- random search
        best, trials = obj.process(fn_name = args.method + '_objective', space=space, 
                              trials=bayes_trials, algo=tuning_algo, max_evals=data_params['max_evals'])
        
        # save trial
        save_obj(bayes_trials,data_params['filename'])
        
    elif data_params['analysis_type'] == 'training':
        if args.method == 'lgb':
            # load parameter
            params = lgb_params(model_params)
            obj = lgb_net(train, train_label, data_params['nfold'], params)
            
            # first training
            train_pred = obj.fit_predict()
            
"""
#%% Additional
train_2 = pd.read_csv('data_npy/train_2.csv')
y_pred = obj.predict(train_2)
y_pred = np.reshape(y_pred, (-1,144))
train_pred = np.reshape(train_pred, (-1,144))

#%% correlation
coef = np.zeros((30,3))
for i in range(30):
    for j in range(3):
        coef[i,j] = np.corrcoef(y_pred[j,:], train_pred[i,:])[1,0]
# coef = np.mean(coef, axis=1)
similar_day = coef>0.97

#%% 해당 day에서 센서값
for i in range(18):
    sensor = train_label.iloc[:,i].values
    sensor = np.reshape(sensor,(30,-1))
    sensor = sensor[[7,22,27],:]
    sensor = np.ravel(sensor)
    Y_18 = pd.read_csv('data_npy/train_label_2.csv')
    Y_18 = np.reshape(Y_18.values, (3,-1))
    Y_18 = np.ravel(Y_18)
    
    print(i,'th correlation is',np.corrcoef(sensor, Y_18)[1,0])
"""
#%%
"""
# test 만들기
test = pd.read_csv('data_raw/test.csv')
test = test.loc[:,'id':'X39']
test['time'] = test.id.values % 144
test = test.drop(columns = 'id')

train = pd.read_csv('data_raw/train.csv')
train = train.loc[:,'id':'X39']
train['time'] = train.id.values % 144
train = train.drop(columns = 'id')
new = pd.concat([train.iloc[-1:,:],test],axis=0).reset_index()
test = add_profile_v2(new,['X00'], N=1)
test = test.drop(columns = 'index')
del test['level_0']
y_pred = obj.predict(test)


ref = pd.read_csv('submit/sample_submission_v7.csv')

mse_AIFrenz(ref.Y18, y_pred)

ref['Y18'] = y_pred
ref.to_csv('submit/submit_3.csv',index=False)

"""
