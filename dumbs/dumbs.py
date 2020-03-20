# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 21:11:43 2020

@author: guseh
"""


    """
    train, train_2, train_label, train_label_2, test, _ = load_dataset('data_raw/')
    train['time'] = train['id'] % 144
    train_2['time'] = train_2['id'] % 144
    test['time'] = test['id'] % 144
    train_label = train_label.drop(columns='id')
    train_label_2 = train_label_2.drop(columns='id')
    
    
    train = np.load('data_npy/train_pred_2.npy')
    _, _, _, train_label, _, submit = load_dataset('data_raw/')
    train_label = train_label.drop(columns='id')
    test = np.load('data_npy/test_pred.npy')
    
    # load parameter
    trials = load_obj('tmp')
    config = dict()
    config['nfold'] = 432
    best = trials.best_trial['result']['params']
    obj = HPOpt_cv(train, train_label.values, config, mse_AIFrenz)
    
    # first training
    obj.lgb_reg(best)
    train_pred = obj.fit_predict()    
    # train_pred_2 = obj.predict(train_2)
    test_pred = obj.predict(test)
    submit.iloc[:,1:] = test_pred
    submit.to_csv('submit/submit_2.csv')
    
    # save
    np.save('data_npy/train_pred.npy', train_pred)
    np.save('data_npy/train_pred_2.npy', train_pred_2)
    np.save('data_npy/test_pred.npy', test_pred)
    
    train_pred_2 = np.load('data_npy/train_pred_2.npy')
    train_pred_2 = train_pred_2[:,0]
    """
    
def main():

    lgb_para = parameter_setting(config)
    obj = HPOpt_cv(train, train_label, config)
    tuning_algo = tpe.suggest # tpe.rand.sugggest -- random search
    lgb_opt = obj.optimize(fn_name='lgb_reg', space=lgb_para, trials=bayes_trials_1, algo=tuning_algo, max_evals=args.max_evals)

def tune_param():

                submit.iloc[:,1:] = test_pred
            submit.to_csv('submit/submit_2.csv')
            
            # save
            np.save('data_npy/train_pred.npy', train_pred)
            np.save('data_npy/train_pred_2.npy', train_pred_2)
            np.save('data_npy/test_pred.npy', test_pred)
            
            train_pred_2 = np.load('data_npy/train_pred_2.npy')
            train_pred_2 = train_pred_2[:,0]

            # train_pred_2 = obj.predict(train_2)
            # test_pred = obj.predict(test)