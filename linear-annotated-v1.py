"""
OLS for intraday volatility forecasting
"""
import argparse

import numpy as np
import pandas as pd
import torch
from os.path import join
import os
import json
import argparse
import matplotlib.pyplot as plt

# ! Args and environment setup ================================================

parser = argparse.ArgumentParser()
parser.add_argument('--back_day',type = int, default= 260)
parser.add_argument('--window_length',type = int,default = 13*60)
parser.add_argument('--train_size',type = int,default = 13*800)
parser.add_argument('--index',type = int,default = 8)
parser.add_argument('--freq',type = str,default = '10min')
parser.add_argument('--market',type=int,default = 0)
args=parser.parse_args()
args.back_day = list(range(args.back_day))

# Save command line args to file
with open('commandline_args%'+str(args.index)+'.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

# Set environment variables
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# get device
device = torch.device("cuda:0")

# ! Data Loading =============================================================

if args.freq == '10min': # 39 10-min intervals in a day
    data = pd.read_csv('hf_dataDec7_stock_10min.csv', index_col = 0)
    args.back_day = 39*20
    args.window_length = 39*250
    args.train_size = 39*1000
elif args.freq == '30min': # 13 30-min intervals in a day
    data = pd.read_csv('hf_dataDec3_stock.csv', index_col=0)
    args.back_day = 13*20
    args.window_length = 13*250
    args.train_size = 13*1000
elif args.freq == '65min': # 6 65-min intervals in a day
    data = pd.read_csv('hf_dataDec7_stock_65min.csv', index_col=0)
    args.back_day = 6*20
    args.window_length = 6*250
    args.train_size = 6*1000
elif args.freq == 'daily': # 1 day
    data = pd.read_csv('hf_dataJan9_stock_daily.csv', index_col=0)
    args.back_day = 1*20
    args.window_length = 1*250
    args.train_size = 1*1000

args.back_day = list(range(args.back_day)) # list of days to look back

# ! Data Preprocessing =======================================================

data = data.fillna(method='ffill') # fill missing values with previous values

namelist = data.columns[:93] # list of stock names
namelist = [x[:-4] for x in namelist] 

# set values above 99.9 percentile to 99.9 percentile, set values below 0.1 percentile to 0.1 percentile
if args.freq != 'daily':
    for clm in data.columns:
        max_p = np.percentile(data[clm], 99.9) # 99.9 percentile
        min_p = np.percentile(data[clm], 0.1) # 0.1 percentile

        data.loc[data[clm] > max_p, clm] = max_p # set values above 99.9 percentile to 99.9 percentile
        data.loc[data[clm] < min_p, clm] = min_p # set values below 0.1 percentile to 0.1 percentile

# 
for ind in namelist:
    #data[ind+'_ret'] = data[ind+'_ret'] *100
    #data[ind+'_vol'] = data[ind+'_vol']*10000
    data[ind + '_logvol'] = np.log(data[ind + '_vol'] + 1e-16) # log of volatility, fuzz with 1e-16 to avoid log(0)

if args.market ==1:
    data['mean_logvol'] = data[data.columns[['logvol' in x for x in data.columns]]].mean(axis=1)

#data.index = pd.to_datetime(data.index, format = '%Y/%m/%d').strftime('%Y-%m-%d')


#data = pd.read_csv('datasector10_4.csv', index_col = 0)
#data.index = pd.to_datetime(data.index, format = '%Y/%m/%d').strftime('%Y-%m-%d')
#namelist  = [x[:-6] for x in data.columns[:105]]

#%%

date = data.index 


class preprocess():
    def __init__(self, input, target, back_day = list(range(0,15)), forward_day = 1):
        # x attribute will hold the predictor variables
        # y attribute will hold the target variable
        # idx attribute will hold the date

        #input is a list of dataframes, for example [price,volatility] with index as the same as target.
        self.x = []
        for df in input:
            # Shift the dataframe by each value in back_day and concatenate along columns
            shifted_df = pd.concat(
                list(map(lambda n: df.shift(n), back_day)), axis=1
            ).reset_index(drop=True).loc[:, ::-1] # Also, reset index and drop to align with the target
            self.x.append(np.expand_dims(np.array(shifted_df), axis=2)) # Expand dimensions to make it compatible for future concatenation
        
        self.x = np.concatenate(tuple(self.x), axis=2) # Concatenate all processed input data along the last axis
        self.idx1 = [~np.any(np.isnan(p)) for p in self.x] # Create an index mask where none of the elements in the x dataframes are NaN
        self.y = target.shift(-forward_day) # Shift the target by forward_day to align with predictor variables
        self.y = pd.DataFrame((self.y)).reset_index(drop=True) # Reset index to align with self.x
        self.idx2 = self.y.notna().all(axis=1) # Create an index mask where none of the elements in the y dataframe are NaN
        self.idx = np.logical_and(self.idx1, self.idx2) # Combine the two index masks
        
        # Filter x and y data based on combined index mask
        self.x = self.x[self.idx]
        self.y = np.array(self.y[self.idx].reset_index(drop=True))

        # Filter date based on combined index mask
        self.idx = data.index[self.idx]


# Winsorize to the 1st and 99th percentile
def normalize(x):
    from scipy.stats.mstats import winsorize
    y = np.empty_like(x) # Create an empty array with the same shape and type as x
    if len(y.shape) == 3: # Handle 3D arrays
        for i in range(x.shape[-1]): # Winsorize each 2D slice along the last axis of the 3D array
            y[:,:,i] = winsorize(x[:,:,i],[0.01,0.01]) # Trim the data at 1% on both the low and high ends of the data distribution
    else: # Handle 2D arrays
        for i in range(x.shape[-1]): # Winsorize each column of the 2D array
            y[:,i] = winsorize(x[:,i],[0.01,0.01]) # Trim the data at 1% on both the low and high ends of the data distribution
    return y


class rolling_predict():
    def __init__(self, keywords = ['XVZ_volatility'], back_day = list(range(0,15)), lr = 0.001): 
        self.back_day = back_day # list of days to look back
        self.lr = lr # learning rate
        self.keywords = keywords # list of keywords
        temp = [data[namelist[0]+'_logvol']] # list of dataframes
        if args.market == 1: # if market is augmented
            temp.append(data['mean'+'_logvol'])
        self.a = preprocess(temp,data[namelist[0]+'_logvol'], back_day = back_day) # preprocess the data

        for ind in namelist[1:]: # for each stock
            temp = []
            for i in [ ind+'_logvol']: # for each keyword

                temp.append(data[i])
                if args.market == 1:
                    temp.append(data['mean' + '_logvol'])

            temp_a = preprocess(temp,data[ind+'_logvol'], back_day = back_day) # preprocess the data
            self.a.x = np.concatenate([self.a.x, temp_a.x],axis = 0 ) # concatenate the x data (predictor variables)
            self.a.y = np.concatenate([self.a.y, temp_a.y],axis = 0 ) # concatenate the y data (target variable)
            self.a.idx = np.concatenate([self.a.idx, temp_a.idx],axis = 0 ) # concatenate the date data

    def train(self, train_index, predict_index,  lr,  names, Epoch_num = 300, pre = True): # train the model

        temp_train_start = np.where(self.a.idx == train_index[0])
        temp_index_train = []
        for i in temp_train_start[0]:
            temp_index_train.extend(list(range(i, i + len(train_index))))
        temp_predict_start = np.where(self.a.idx == predict_index[0])
        temp_index_predict = []
        for i in temp_predict_start[0]:
            temp_index_predict.extend(list(range(i, i + len(predict_index))))
        train_x = self.a.x[temp_index_train]
        train_y = self.a.y[temp_index_train]
        test_x = self.a.x[temp_index_predict]
        test_y = self.a.y[temp_index_predict]

        train_x = train_x.reshape(train_x.shape[0], -1)
        test_x = test_x.reshape(test_x.shape[0], -1)
        train_x = np.concatenate((np.ones((train_x.shape[0], 1)), train_x), axis=1)
        test_x = np.concatenate((np.ones((test_x.shape[0], 1)), test_x), axis=1)

        beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(train_x.T, train_x)), train_x.T), train_y)
        predict = np.matmul(test_x, beta)

        predict = np.reshape(predict, (-1, len(namelist)), 'F')
        test_y = np.reshape(test_y, (-1, len(namelist)), 'F')
        plot_valid = np.concatenate((predict, test_y), axis=1)
        plot_valid = pd.DataFrame(plot_valid)
        plot_valid.index = date[(np.where(date == predict_index[0])[0][0] + 1):(
                np.where(date == predict_index[-1])[0][0] + 2)]
        plot_valid.columns = [x + 'out' for x in namelist] + [x + 'real' for x in namelist]
        return plot_valid


    def run(self, window_length, train_size, Epoch_num = 2, pre = True):
        T = int(self.a.x.shape[0]/len(namelist))
        result_list = []
        if args.freq != 'daily':
            start_index = np.where(self.a.idx == '2015-06-30'+'/'+'16:00')[0][0]
        else:
            start_index = np.where(self.a.idx == '2015-06-30')[0][0]
        for start in range(start_index,T-1, window_length):
            print(self.a.idx[start])
            if start + window_length <= T - 1:
                result_list.append(
                    self.train(Epoch_num=Epoch_num, train_index=self.a.idx[start_index - train_size:start],
                               predict_index=self.a.idx[start:start + window_length], lr=None, names=None, pre=pre))
            else:
                result_list.append(
                    self.train(Epoch_num=Epoch_num, train_index=self.a.idx[start_index - train_size:start],
                               predict_index=self.a.idx[start: T - 1], lr=None, names=None, pre=pre))
        return result_list


q = rolling_predict( back_day= args.back_day,
            lr=0.001)
result = q.run(args.window_length,args.train_size,Epoch_num = 2,pre = False)
MYDIR = 'hf_'+args.freq+'/linear'
CHECK_FOLDER = os.path.isdir(MYDIR)
if not CHECK_FOLDER:
    os.makedirs(MYDIR)
if args.market == 0:
    pd.concat(result).to_csv(MYDIR+'/meta.csv')
elif  args.market == 1:
    pd.concat(result).to_csv(MYDIR+'/aug.csv')
'''
    q = rolling_predict(keywords=['SPY_volatility','SPY_ret'], back_day=list(range(0, 15)),
                lr=0.001)
    result = q.run(30,1000,Epoch_num = 20000,pre = False)
    pd.concat(result).to_csv('LSTM_SPY3_2.csv')
'''

if args.market == 0:
    from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_percentage_error
    if args.freq != 'daily':
        lstm_result = pd.read_csv(MYDIR+'/meta.csv',index_col=0).loc['2015-07-01/09:30':]
    else:
        lstm_result = pd.read_csv(MYDIR+'/meta.csv',index_col=0).loc['2015-07-01':]
    report_df = pd.DataFrame(index = namelist,columns=['MSE','r2_score'])
    for i in namelist:
        report_df.loc[i,'MSE'] = mean_squared_error(lstm_result[i+'out'],lstm_result[i+'real'])
        report_df.loc[i,'r2_score'] = r2_score( lstm_result[i + 'real'],lstm_result[i + 'out'])
        report_df.loc[i,'MAPE'] = mean_absolute_percentage_error( lstm_result[i + 'real'],lstm_result[i + 'out'])
    report_df.to_csv(MYDIR+'/meta_report.csv')
elif  args.market == 1:
    from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_percentage_error
    if args.freq != 'daily':
        lstm_result = pd.read_csv(MYDIR+'/aug.csv',index_col=0).loc['2015-07-01/09:30':]
    else:
        lstm_result = pd.read_csv(MYDIR+'/aug.csv',index_col=0).loc['2015-07-01':]
    report_df = pd.DataFrame(index = namelist,columns=['MSE','r2_score'])
    for i in namelist:
        report_df.loc[i,'MSE'] = mean_squared_error(lstm_result[i+'out'],lstm_result[i+'real'])
        report_df.loc[i,'r2_score'] = r2_score( lstm_result[i + 'real'],lstm_result[i + 'out'])
        report_df.loc[i,'MAPE'] = mean_absolute_percentage_error( lstm_result[i + 'real'],lstm_result[i + 'out'])
    report_df.to_csv(MYDIR+'/aug_report.csv')
