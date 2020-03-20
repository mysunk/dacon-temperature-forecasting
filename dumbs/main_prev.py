
# =============================================================================
# Learn train_1 by cv result
# =============================================================================

# load cv result
trials = load_obj('rf_10fold_2')
print(trials.best_trial['result']['params'])

# example
param = {'max_depth':180,
    'max_features':26,
    'n_estimators':370,
    'random_state': 0
    }

# load raw data
train_1, train_2, train_label_1, train_label_2, test, submit = load_dataset('data_raw/')
train_1['time'] = train_1['id'] % 144
train_2['time'] = train_2['id'] % 144
test['time'] = test['id'] % 144

train_label_1 = train_label_1.drop(columns='id')
train_label_2 = train_label_2.drop(columns='id')

# train-test split
nfold=10
skf = KFold(n_splits=nfold, random_state=None, shuffle=False)

oof_loss = np.zeros((nfold,2)) # train loss and val loss
oof_pred = np.zeros(train_label_1.shape)
oob_pred = [[np.zeros((test.shape[0],train_label_1.shape[1]))]]
models = []

train = train_1
train_label = train_label_1

# First train phase
for i, (train_index, test_index) in enumerate(skf.split(train, train_label)):
    model = RandomForestRegressor(**param)
    x_train = train.iloc[train_index]
    y_train = train_label.iloc[train_index]
    x_test = train.iloc[test_index]
    y_test = train_label.iloc[test_index]
    
    # make classifier
    model.fit(x_train, y_train)
    y_pred_test = model.predict(x_test)
    y_pred_train = model.predict(x_train)
    
    # cal loss
    oof_loss[i] = (mse_AIFrenz(y_train, y_pred_train),mse_AIFrenz(y_test, y_pred_test)) # 순서대로 train, test loss
    
    # oof prediction
    oof_pred[test_index,:] = y_pred_test
    models.append(model)
    
    # oob prediction
    oob_pred.append(model.predict(test))
oob_pred = np.mean(oob_pred,axis=1)

# =============================================================================
# predict train_2
# =============================================================================

oof_pred = []
# Second train phase
for i in range(nfold):
    # make classifier
    model = models[i]
    y_pred = model.predict(train_2)
    oof_pred.append(y_pred)
oof_pred = np.mean(oof_pred,axis=0)
np.save('result/train_2_pred',oof_pred)

# =============================================================================
# Apply stack result
# =============================================================================
trials = load_obj('eln_144fold_1')
print(trials.best_trial['result']['params'])

# load dataset
train = np.load('result/train_2_pred.npy')

# example
param = {'alpha':0.013448014319253621,
    'l1_ratio':0.7000000000000001,
    'max_iter':10000,
    'random_state': 0
    }

# train model
model =  ElasticNet(**param)
model.fit(train, train_label_2)
result = model.predict(oob_pred)
submit.iloc[:,1] = result
submit.to_csv('submit/submit_2.csv',index=False)
