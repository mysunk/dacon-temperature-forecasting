# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 22:25:48 2020

@author: guseh
"""
from util import *
import pandas as pd
import matplotlib.pyplot as plt

train_1, train_2, train_label_1, train_label_2, test, sample = load_dataset('data_raw/')

#%% 하루 단위로
def plot_feature(data, feature_name, days):
    plt.figure()
    for i in days:
        timestamp = range(i*24*6,(i+1)*24*6)
        plt.plot(timestamp, data[feature_name][list(timestamp)])
    plt.legend(days)
    plt.xlabel('time')
    plt.title(feature_name)
    
def plot_feature_2(data, feature_name, days):
    plt.figure()
    for i in days:
        timestamp = range(i*24*6,(i+1)*24*6)
        plt.plot(range(24*6), data[feature_name][list(timestamp)])
    plt.legend(days)
    plt.xlabel('time')
    plt.title(feature_name)

def plot_features(data, features, days):
    plt.figure()
    timestamp = range(days[0]*24*6,days[-1]*24*6)
    for i in features:
        plt.plot(timestamp, data[i][list(timestamp)])
    plt.legend(features)
    plt.xlabel('time')

#%%
plot_feature(train_label_1, 'Y02',range(30))
plot_feature(train_label_2, 'Y18',range(30,32))
plot_feature_2(train_1, 'X00',range(2))
plot_feature_2(train_1, 'X01',range(2))
plot_features(train_2, ['X00','X07','X28','X31','X32'], range(3))
plot_features(train_label_2, train_label_1.columns[1:10], range(5))

#%%
test = pd.read_csv('../data_raw/test.csv')
plt.plot(irradiance_difference(test['X11'].values))


#%% 모든 feature들 n일간
for feature in train_1.columns[1:]:
    plot_feature(train_1,feature ,range(0,30))

#%% 
for feature in train_label_1.columns[1:]:
    plot_feature(train_label_1,feature ,range(2))

#%% Y18 확인

plt.figure()
plt.plot(train_label_2.Y18)

#%% test 확인

for feature in test.columns[1:]:
    plot_feature(test,feature ,range(0,3))
    
#%% For setting
from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'qt')
get_ipython().run_line_magic('matplotlib', 'inline')

#%% Seasonality removal
from statsmodels.tsa.seasonal import seasonal_decompose

seasonal_decompose(train_1, period = 6*24*).seasonal

#%% correlation
train_label_1 = train_label_1.drop(columns='id')
columns = train_label_1.columns
corr = train_label_1[columns].corr()

#%% load pkl result
from util import *
trials = load_obj('Y16_N1')
best = trials.best_trial['result']['params']
print(best)

#%%
import pandas as pd
train_1 = pd.read_csv('data_npy/train_1.csv')
train_2 = pd.read_csv('data_npy/train_2.csv')
train_label_1 = pd.read_csv('data_npy/train_label_1.csv')
train_label_2 = pd.read_csv('data_npy/train_label_2.csv')
test = pd.read_csv('data_npy/test.csv')

#%% 온도
plot_feature_2(train_1, 'X09',range(3))

#%%
a = [1,2,3]

#%% 온도
plot_features(train_1, ['X00','X07','X28','X31','X32'], range(4))
plot_features(train_2, ['X00','X07','X28','X31','X32'], range(4))
plot_features(test, ['X00','X07','X28','X31','X32'], range(4))

#%% label
plot_features(train_label_1, ['Y16'], range(4))
plot_features(train_label_2, ['Y18'], range(4))

#%% similarity
similarity = np.zeros((30,3))
for i in range(30):
    for j in range(3):
        similarity[i,j] = np.linalg.norm(train.iloc[i].values-test.iloc[j].values)

#%%
import matplotlib.pyplot as plt
plt.plot(train_label_1[:2000])
# plt.plot(train_label_1[:2000])
plt.plot(train_label_1_ref.loc[:2000,'Y16'])

# plt.figure()
plt.plot(train_label_2)

#%% correlation
import numpy as np
train_1['solar_diff_X11'] = difference(train_1.X11.values)
corr = np.corrcoef(np.vstack([np.ravel(train_label_1_ref.Y16.values), train_1.loc[:,'solar_diff_X11'].values]))
corr = np.corrcoef(np.vstack([np.ravel(train_label_1_ref.Y16.values), train_1.loc[:,'X11'].values]))

#%%
ref = pd.read_csv('../submit/sample_submission_v22.csv')
mse_AIFrenz(ref.Y18.values, y_pred)

#%%
interv = range(5000)
plt.plot(ref.Y18.values[interv])
plt.plot(trial[interv])
plt.plot(submit_3.Y18.values[interv])
plt.plot(test.X00.values[interv])
plt.legend(['sw','ensemble','prev_submit','X00'])
#%% 과거 제출값
submit_3 = pd.read_csv('../submit/submit_3.csv')
submit_3.Y18 = trial
submit_3.to_csv('submit/submit_4.csv',index=False)
#%%
y_pred_2 = y_pred
#%%
mean_val = np.mean([y_pred_1, y_pred_2],axis=0)

#%%
from sklearn.metrics import mean_squared_error
print('diff with sw is',mean_squared_error(ref.Y18.values, trial))
print('diff with sw is',mse_AIFrenz(ref.Y18.values, trial))

#%%
trial = y_pred * 0.7 + ref.Y18.values * 0.3

#%%
plt.plot(train.X00[:1000])
plt.plot(train_label_ref.Y16[:1000])

#%%
from util import *
import pandas as pd
import matplotlib.pyplot as plt

interv = range(1000)
plt.plot(ref.Y18.values[interv])
plt.plot(y_pred[interv])
plt.legend(['sw','ms'])

#%% w
from util import *
import pandas as pd
import matplotlib.pyplot as plt

ref = pd.read_csv('../submit/sample_submission_v26.csv')
test = pd.read_csv('../data_raw/test.csv')

interv = range(300)
plt.plot(ref.Y18.values[interv])
plt.plot(test.X00.values[interv],'--')
plt.plot(test.X11.values[interv],'--')
plt.plot(test.X12.values[interv],'--')
plt.legend(['sw','T','S','H'])

#%% 3일치
Y18 = pd.read_csv('data_npy/Y_18.csv')
train = pd.read_csv('../data_raw/train.csv')
train = train.loc[:,'id':'X39']
interv = range(30*144,33*144)
plt.plot(Y18.Y18.values[interv])
plt.plot(train.X00.values[interv],'--')
plt.plot(train.X11.values[interv],'--')
plt.plot(train.X12.values[interv],'--')
plt.legend(['ground_truth','T','S','H'])

#%% 일일 누적 일사량 -> 그냥 일사량으로
import numpy as np


tmp = difference(train.X34.values)
tmp2 = train.X34.values
plt.plot(tmp)

#%%
plt.plot()

#%%
from util import *
import matplotlib.pyplot as plt
_, _, _, train_label_2, _,_= load_dataset('data_raw/')
Y_18 = pd.read_csv('data_npy/Y_18.csv')
plt.figure()
plt.plot(train_label_1.loc[range(1000),['Y15','Y16']],'--')
plt.plot(Y_18.values[range(1000)])
plt.legend(['Y15','Y16','Y_18_tmp'])

#%% 15, 22, 27일 plot
train_2 = train_2.reset_index(drop=True)
plt.figure()
plt.plot(train_1.loc[range(27*144,28*144),'X06'].reset_index(drop=True))
plt.plot(train_2.loc[range(144),'X06'])

#%%
mean_val = np.mean(train_label_1.loc[:,['Y15','Y16']].values,axis=1)
mean_val = np.hstack([mean_val, train_label_2.Y18])
Y_18.loc[:,'Y18'] = mean_val
plt.plot(train_label_1.loc[:,'Y15'])
plt.plot(Y_18.values)
plt.legend(['Y09','Y18_tmp'])

Y_18.to_csv('data_npy/Y_18_trial_1.csv',index=False)

#%% 3일치 plot
plt.plot(train_label_2.Y18)

#%%
Y_18 = pd.read_csv('data_npy/Y_18_trial_1.csv')
plt.figure()
plt.plot(Y_18.iloc[range(29*144,30*144)])
# plt.plot(train_label_2.Y18)

#%%

plot_features(train_label_1,['Y15','Y16'],range(5))
#%%
train_2 = train_2.reset_index(drop=True)
train_label_2 = train_label_2.reset_index(drop=True)
interv = range(144)
plt.plot(train_2.X00.values[interv])
# plt.plot(train_2.X01.values[interv])
# plt.plot(train_2.X02.values[interv])
# plt.plot(train_2.X03.values[interv])
# plt.plot(train_2.X04.values[interv])
# plt.plot(train_2.X05.values[interv])
plt.plot(train_2.solar_diff_X11.values[range(144)]*50)
train_1['solar_diff_X11'] = irradiance_difference(train_1.X11.values)
train_2['solar_diff_X11'] = irradiance_difference(train_2.X11.values)
train_1['solar_diff_X34'] = irradiance_difference(train_1.X34.values)
train_2['solar_diff_X34'] = irradiance_difference(train_2.X34.values)
plt.plot(train_label_2.Y18.values[range(144)])
plt.legend(['X00','X02','X03','X11_diff','Y18'])

np.corrcoef(train_2.X00, train_2.solar_diff_X11)

#%%
import matplotlib.pyplot as plt
plt.plot(y_pred)
plt.plot(train_label_2.Y18.values)

#%%
import matplotlib.pyplot as plt
interest = 'Y15'
plt.plot(y_pred)
plt.plot(train_label_1.loc[:,interest])
plt.legend([interest,'Y18_tmp'])

#%%
for day in range(3):
    plt.figure()
    plt.plot(train_1.X01.loc[range(144*day,144*(day+1))].reset_index(drop=True))
    plt.plot(train_2.X01.loc[range(144)].reset_index(drop=True))
    plt.legend(['train','test'])
    plt.title('day'+str(day))
    
#%% similarity 계산

train_2 = train_2.reset_index(drop=True)
train_1 = train_1.reset_index(drop=True)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_1.loc[:,:] = scaler.fit_transform(train_1.loc[:,:].values)
train_2.loc[:,:] = scaler.transform(train_2.loc[:,:].values)
#%%
feature = ['X00','X07','X11','X12','X15','X20','X30','X31','X32','X34','X37','X38','solar_diff_X11','solar_diff_X34']
feature_weight = np.array([3,3,1,0,0,0,0,3,3,0,0,0,1,1])
feature_weight = feature_weight / np.sum(feature_weight)
simday_index = np.ones((144,3)) # day 1, 2, 3
for point in range(144): # 한 포인트 단위로
    for day in range(3):
        similarity = np.linalg.norm((np.vstack([train_2.loc[day*144+point,feature].values]*30)-train_1.loc[range(point,4320,144),feature].values)*feature_weight,axis=1)
        print('for',point,'th iteration, min value is',min(similarity))
        simday_index[point,day] = np.argmin(similarity)
for i in range(432):
    train_label_2.loc[i,train_label_1.columns[1:]] = train_label_1.loc[(int)(simday_index[i%144,(int)(i/144)])*144+(i%144),train_label_1.columns[1:]]
    



#%%
train_label_2 = train_label_2.reset_index(drop=True)
for i in range(1,19):
    train_label_2[train_label_1.columns[i]] = np.zeros((432))


#%%
plt.plot(train_label_2.Y15.values)
plt.plot(train_label_2.Y18.values)
# plt.plot((train_label_2.Y15.values+train_label_2.Y16.values)/2)
# plt.plot(train_label_1.Y16.values[range(432)])
# plt.plot(ref.Y18.values)
# plt.plot(train_label_1.Y15.values[range(432)])
# ref = pd.read_csv('data_npy/Y_18.csv')

#%%

    start_h = point * 6
    end_h = start_h + 5
    for i in range(6):
        similarity[point] = np.linalg.norm(np.vstack([train_2.loc[point,feature].values]*4320)-train_1.loc[start,feature].values,axis=1)

#%% moving average
data = train_label_2.Y16.values
new_2 = data
for i in range(1,430-1):
    new_2[i] = np.mean(data[i-1:i+1])

    
#%%
plt.plot(train_label_2.Y18.values)
# plt.plot(train_label_2.Y15.values,'--')
# plt.plot(train_label_2.Y16.values,'--')
new = train_label_2.Y16.values*0.75 + train_label_2.Y15.values*0.36
plt.plot(new,'--')
# plt.plot(new_2*1.3 - new_1*0.2)

#%%

new = train_label_1.Y16.values * 0.75 + train_label_1.Y15.values * 0.36
plt.plot(new)
plt.plot(train_label_1.Y16.values)
plt.plot(range(4320,4320+432),train_label_2.Y18.values)
plt.plot(ref.Y18.values[range(4320)],'--')
plt.legend(['new','Y16','Y18','sw'])

ref.iloc[:4320,0] = new
ref.to_csv('data_npy/Y_18_w.csv',index=False)

#%%
import matplotlib.pyplot as plt
plt.plot(y_pred)
plt.plot(train_label_2.Y18.values)

#%%
Y16_pred = np.mean(preds_Y16, axis=0)
Y15_pred = np.mean(preds_Y15, axis=0)

tmp = train_label_2
tmp['Y18'] = Y16_pred
tmp.to_csv('data_npy/Y16_pred.csv',index=False)
train_label_2.to_csv('data_npy/Y18_ref.csv')

#%%
for i in range(5):
    plt.plot(preds_Y16[i],':')

#%%
interv = range(144)
plt.plot(Y15_pred[interv])
plt.plot(Y16_pred[interv])
plt.plot(train_label_2.Y18.values[interv])
plt.legend(['Y15','Y16','Y18'])

#%%
train_label_1.to_csv('data_npy/train_label_2.csv',index=False)

#%%
Y18_ms = pd.read_csv('data_npy/Y18_ms.csv',header=None)
Y18_sw = pd.read_csv('data_npy/Y_18_1.csv')
tmp = pd.read_csv('data_npy/Y_18_2.csv')
Y18_sw['Y18_2'] = tmp.Y18.values
#%%
ref = pd.read_csv('../submit/sample_submission_v26.csv')
mse_AIFrenz(ref.Y18.values, y_pred_1)
mean_squared_error(ref.Y18.values, y_pred_1)
y_pred = np.mean(preds,axis=0)
plt.plot(y_pred)

#%%
plt.plot(y_pred)
plt.plot(ref.Y18.values)

#%%
y_pred_15 = np.mean(preds,axis=0)
y_pred_16 = np.mean(preds,axis=0)

#%%
interv = range(6000,7000)
plt.plot(y_pred_1[interv])
# plt.plot(y_pred_2[interv])
plt.plot(trial[interv])
plt.plot(ref.Y18.values[interv],'--')
plt.legend(['ms1','ms2','sw'])

#%%
trial = y_pred_1 * 0.4 + y_pred_2 * 0.05 + y_pred_3 * 0.05 + ref.Y18.values * 0.5
#%%
res_y15= pd.read_csv('data_npy/res_15.csv',header=None)
res_y16= pd.read_csv('data_npy/res_16.csv',header=None)
y18_w_y15 = y_pred_15 + np.ravel(res_y15.values)
y18_w_y16 = y_pred_16 + np.ravel(res_y16.values)
y_pred_1 = y18_w_y15 * 0.5 + y18_w_y16 * 0.5
mse_AIFrenz(ref.Y18.values, y_pred_1)
#%%
mean_res_y15= pd.read_csv('data_npy/mean_res_y15.csv',header=None)
mean_res_y16= pd.read_csv('data_npy/mean_res_y16.csv',header=None)
y18_w_y15 = y_pred_15 + np.mean(mean_res_y15.values)
y18_w_y16 = y_pred_16 + np.mean(mean_res_y16.values)
y_pred_2 = y18_w_y15 * 0.5 + y18_w_y16 * 0.5
mse_AIFrenz(ref.Y18.values, y_pred_2)
#%%
mean_res_y15= pd.read_csv('data_npy/mean_res_y15.csv',header=None)
mean_res_y16= pd.read_csv('data_npy/mean_res_y16.csv',header=None)
y18_w_y15 = y_pred_15 + np.ravel(mean_res_y15.values)
y18_w_y16 = y_pred_16 + np.ravel(mean_res_y16.values)
y_pred_3 = y18_w_y15 * 0.5 + y18_w_y16 * 0.5
mse_AIFrenz(ref.Y18.values, y_pred_3)
#%%

#%%



#%%
ref.Y18 = y_pred_1
ref.to_csv('submit/submit_5.csv',index=False)

ref.Y18 = trial
ref.to_csv('submit/submit_6.csv',index=False)
#%%
import numpy as np
from util import *
import pandas as pd
import matplotlib.pyplot as plt
Y15_pred = np.load('data_npy/Y15_pred.npy')
train_1, train_2, train_label_1, train_label_2, test, sample = load_dataset('data_raw/')
train_label_2 = train_label_2.Y18.values

#%%
import numpy as np
from util import *
import pandas as pd
import matplotlib.pyplot as plt
Y16_pred = np.load('data_npy/Y16_pred.npy')
train_1, train_2, train_label_1, train_label_2, test, sample = load_dataset('data_raw/')
train_label_2 = train_label_2.Y18.values

#%%
# plt.plot(Y15_pred)
# plt.plot(train_label_2)
plt.plot()
train = train.loc[:,['time','X00','X07','X30','X31','X34','X34_diff']]

#%%

train_2 = train_2.reset_index(drop=True)

scaler = StandardScaler()
train_2.loc[:,:] = scaler.fit_transform(train_2.values)

#%%
res = train_label_2-Y15_pred
day1 = res[:144]
day2 = res[144:288]
day3 = res[288:]
plt.plot(day1,'--')
plt.plot(day2,'--')
plt.plot(day3,'--')
mean_val = (day1 + day2 + day3)/3
plt.plot(mean_val)

#%%
res = train_label_2-Y16_pred
day1 = res[:144]
day2 = res[144:288]
day3 = res[288:]
plt.plot(day1,'--')
plt.plot(day2,'--')
plt.plot(day3,'--')
mean_val = (day1 + day2 + day3)/3
plt.plot(mean_val)

#%%
plt.plot(train_2.X00[:144])
plt.plot(train_2.X07[:144])
plt.plot(train_2.X02[:144])

#%%
train_label_2.to_csv('matlab/Y18.csv')
train_label_2.Y18 = Y15_pred
train_label_2.to_csv('matlab/Y15.csv')
train_label_2.Y18 = Y16_pred
train_label_2.to_csv('matlab/Y16.csv')

#%%
import numpy as np
from util import *
import pandas as pd
import matplotlib.pyplot as plt
Y15_pred = np.load('data_npy/Y15_pred_80day.npy')
Y16_pred = np.load('data_npy/Y16_pred_80day.npy')

Y15_res = pd.read_csv('../matlab/res_15.csv', header=None)
Y16_res = pd.read_csv('../matlab/res_16.csv', header=None)

Y15_res = np.ravel(Y15_res.values)
Y16_res = np.ravel(Y16_res.values)

Y18_pred = (Y15_pred + 1.7) * 0.5 + (Y16_pred + 1.7) * 0.5

ref = pd.read_csv('../submit/sample_submission_v33.csv')


mse_AIFrenz(Y18_pred,ref.Y18.values)

#%%
interv = range(5000,6000)
plt.plot(Y18_pred[interv])
plt.plot(ref.Y18.values[interv])
# plt.plot(ref_ms.Y18.values[interv])
plt.legend(['now','sw','ms_prev'])

#%%
ref_ms = pd.read_csv('../submit/submit_5.csv')

#%%
ref.Y18 = Y18_pred
ref.to_csv('submit/submit_6.csv',index=False)
