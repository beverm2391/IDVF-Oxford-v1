"""
Train DNN
"""
import argparse

import numpy as np
import pandas as pd
from os.path import join
import os
import json
import argparse
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import torch

# ! Args and config =====================================================

parser = argparse.ArgumentParser()
parser.add_argument('--back_day',type = int, default= 260, help="look-back trading days")
parser.add_argument('--window_length',type = int,default = 13*60, help="number of bins in testing period")
parser.add_argument('--train_size',type = int,default = 13*800, help="number of bins in training period")
parser.add_argument('--index',type = int,default = 8, help="model index for ensemble")
parser.add_argument('--freq',type = str,default = 'daily', help="horizon for computing vol")
parser.add_argument('--count_one_day',type = int,default = 1, help="number of bins in one day")
parser.add_argument('--cuda',type = int,default = 0, help="if GPU")
parser.add_argument('--market',type=int,default = 1, help="if including market vol")
args=parser.parse_args()
args.back_day = list(range(args.back_day))

# save args
with open('commandline_args%'+str(args.index)+'.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

# environ config
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# device
device = torch.device("cuda:"+str(args.cuda))

# ! Load data =====================================================

if args.freq == '10min':
    data = pd.read_csv('hf_data_stock_10min.csv', index_col = 0)
    args.back_day = 39*20
    args.window_length = 39*250
    args.train_size = 39*1000
    args.count_one_day = 39
elif args.freq == '30min':
    data = pd.read_csv('hf_data_stock_30min.csv', index_col=0)
    args.back_day = 13*20
    args.window_length = 13*250
    args.train_size = 13*1000
    args.count_one_day = 13
elif args.freq == '65min':
    data = pd.read_csv('hf_data_stock_65min.csv', index_col=0)
    args.back_day = 6*20
    args.window_length = 6*250
    args.train_size = 6*1000
    args.count_one_day = 6
elif args.freq == 'daily':
    data = pd.read_csv('hf_data_stock_daily.csv', index_col=0)
    args.back_day = 1*20
    args.window_length = 1*250
    args.train_size = 1*1000
    args.count_one_day = 1

args.back_day = list(range(args.back_day))

# clean the data 
data = data.fillna(method='ffill')
namelist = data.columns[:93]
namelist = [x[:-4] for x in namelist]

if args.freq!='daily':
    for clm in data.columns:
        max_p = np.percentile(data[clm], 99.9)
        min_p = np.percentile(data[clm], 0.1)

        data.loc[data[clm] > max_p, clm] = max_p
        data.loc[data[clm] < min_p, clm] = min_p

# logvol
for ind in namelist:
    data[ind+'_logvol'] = (np.log(data[ind+'_vol']+1e-16))

# market vol
if args.market ==1:
    data['mean_logvol'] = data[data.columns[['logvol' in x for x in data.columns]]].mean(axis=1)

date = data.index

if args.market == 0:
    MYDIR = 'hf_'+args.freq+'/dnn/models_meta'+str(args.index)
    CHECK_FOLDER = os.path.isdir(MYDIR)
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)

    save_path = MYDIR
elif args.market == 1:
    MYDIR = 'hf_' + args.freq + '/dnn/models_aug'+str(args.index)
    CHECK_FOLDER = os.path.isdir(MYDIR)
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)

    save_path = MYDIR

# name of model
model_name = 'checkpoint'+str(args.cuda)+'.pt'

# DNN model
class DNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DNN, self).__init__()

        # using linear and leaky relu with optional batch normalization, default is True
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # define the model
        self.model = nn.Sequential(
            *block(input_size, 128), 
            *block(128, 32), 
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        forecast_y = self.model(x)
        return forecast_y


class preprocess():
    def __init__(self, input, target, back_day = list(range(0,15)), forward_day = 1):
        #input is a list of dataframes, for example [price,volatility] with index as the same as target.
        self.x = []
        for _ in input:
            self.x.append(np.expand_dims(np.array(pd.concat(list(map(lambda n: _.shift(n), back_day)), axis=1).reset_index(drop=True).loc[:,::-1]),axis =2))
        self.x = np.concatenate(tuple(self.x),axis =2)
        self.idx1 = [~np.any(np.isnan(p)) for p in self.x]
        self.y = target.shift(-forward_day)
        self.y = pd.DataFrame((self.y)).reset_index(drop=True)
        self.idx2 = self.y.notna().all(axis = 1)
        self.idx = np.logical_and(self.idx1, self.idx2)
        self.x = self.x[self.idx]
        self.y = np.array(self.y[self.idx].reset_index(drop = True))

        self.idx = data.index[self.idx]

# winsorize to the 99% and 1% quantile
def normalize(x):
    from scipy.stats.mstats import winsorize
    y = np.empty_like(x)
    if len(y.shape) == 3:
        for i in range(x.shape[-1]):
            y[:,:,i] = winsorize(x[:,:,i],[0.01,0.01])
    else:
        for i in range(x.shape[-1]):
            y[:,i] = winsorize(x[:,i],[0.01,0.01])
    return y


class mean_var():
    # calculate the mean and variance of the data for normalization
    def __init__(self, data):
        self.ave = np.mean(data, axis = 0)
        self.var = np.var(data,axis = 0)
    # normalize the data
    def preprocess(self, temp):
        return (temp - self.ave)/np.sqrt(self.var)
    # denormalize the data
    def back(self, temp):
        return temp * np.sqrt(self.var)+self.ave


class rolling_predict():
    def __init__(self, keywords = ['XVZ_volatility'], back_day = list(range(0,15)), lr = 0.001):
        self.back_day = back_day
        self.lr = lr
        self.keywords = keywords
        temp = [data[namelist[0] + '_logvol']]
        if args.market == 1:
            temp.append(data['mean' + '_logvol'])
        self.a = preprocess(temp, data[namelist[0] + '_logvol'], back_day=back_day) # initialize the data with the first stock

        for ind in namelist[1:]: # concatenate the data of other stocks
            temp = []
            for i in [ind + '_logvol']:

                temp.append(data[i])
                if args.market == 1:
                    temp.append(data['mean' + '_logvol'])
            temp_a = preprocess(temp, data[ind + '_logvol'], back_day=back_day) 
            self.a.x = np.concatenate([self.a.x, temp_a.x], axis=0)
            self.a.y = np.concatenate([self.a.y, temp_a.y], axis=0)
            self.a.idx = np.concatenate([self.a.idx, temp_a.idx], axis=0)


    def train(self, train_index, predict_index,  lr,  names, Epoch_num = 300, pre = True):
        temp_train_start = np.where(self.a.idx == train_index[0]) # find the index of the first day in the training period
        temp_index_train = []
        for i in temp_train_start[0]:
            temp_index_train.extend(list(range(i, i + len(train_index) - 250 * args.count_one_day))) # get all starting indices of the training set

        temp_index_valid = []
        for i in temp_train_start[0]:
            temp_index_valid.extend(list(range(i + len(train_index) - 250 * args.count_one_day, i + len(train_index)))) # get all starting indices of the validation set

        temp_predict_start = np.where(self.a.idx == predict_index[0])
        temp_index_predict = []
        for i in temp_predict_start[0]:
            temp_index_predict.extend(list(range(i, i + len(predict_index)))) # get all starting indices of the testing set

        train_x = self.a.x[temp_index_train] # get the training inputs
        train_y = self.a.y[temp_index_train] # get the training targets
        valid_x = self.a.x[temp_index_valid] # get the validation inputs
        valid_y = self.a.y[temp_index_valid] # get the validation targets
        test_x = self.a.x[temp_index_predict] # get the testing inputs
        test_y = self.a.y[temp_index_predict] # get the testing targets

        x_stats = mean_var(train_x) # initialize the normalizer
        y_stats = mean_var(train_y) # initialize the normalizer

        train_x = x_stats.preprocess(train_x) # normalize the training inputs
        train_y = y_stats.preprocess(train_y) # normalize the training targets

        valid_x = x_stats.preprocess(valid_x) # normalize the validation inputs
        valid_y = y_stats.preprocess(valid_y) # normalize the validation targets

        test_x  = x_stats.preprocess(test_x) # normalize the testing inputs
        #test_y  = y_stats.preprocess(test_y)

        train_x = train_x.reshape(train_x.shape[0], -1) # reshape the training inputs
        test_x = test_x.reshape(test_x.shape[0], -1) # reshape the testing inputs
        valid_x = valid_x.reshape(valid_x.shape[0], -1) # reshape the validation inputs

        from pytorchtools import EarlyStopping


        if lr is None:
            lr = self.lr
        if names is None:
            names = self.keywords

        # ! Notice the shuffle is True here
        # training dataloader
        trainloader = DataLoader(TensorDataset(torch.tensor(train_x), torch.tensor(train_y)),1024,
                                 shuffle=True) # initialize the training dataloader, batch size is 1024
        
        # ! but the shuffle is False here
        # val dataloader
        validloader1 = DataLoader(TensorDataset(torch.tensor(valid_x), torch.tensor(valid_y)),
                                  torch.tensor(valid_x).shape[0],
                                  shuffle=False) # initialize the validation dataloader, batch size is the number of validation samples
        
        # ! and also False here
        # test dataloader
        validloader2 = DataLoader(
            TensorDataset(torch.tensor(test_x), torch.tensor(test_y)),
            len(torch.tensor(test_y)), shuffle=False) # initialize the testing dataloader, batch size is the number of testing samples

        early_stopping = EarlyStopping(patience=10, verbose=False, path=model_name) # set the early stopping criteria
        net = DNN(train_x.shape[1], 1).to(device) # initialize the model, input dims is the number of features, output dims is 1
        loss_function = nn.MSELoss() # loss function is MSE
        optimiser = optim.Adam(net.parameters(), lr=lr, eps=1e-8) # optimizer is Adam
        Epoch_num = Epoch_num # not sure why this is here

        for epoch in range(Epoch_num):
            net.train() # set the model to train mode
            for data_val, target in trainloader:
                optimiser.zero_grad() # clear the gradients
                output = net(data_val.float().to(device)) # forward pass
                loss = loss_function(output.float().view(-1), target.float().view(-1).to(device)) # compute the loss (flattten the output and target)
                loss.backward() # backprop
                optimiser.step() # update the weights
            net.eval() # set the model to eval mode
            for data_val, target in validloader1:
                output = net(data_val.float().to(device)) # forward pass
                loss_valid = loss_function(output.float().view(-1), target.float().view(-1).to(device)) # compute the validation loss

            # valid_list.append(loss_valid.float().view(-1).detach().cpu().numpy()[0])
            # test_list.append(loss_test.float().view(-1).detach().cpu().numpy()[0])
            early_stopping(loss_valid.detach().cpu().numpy().reshape(-1)[0], net) # check if the validation loss is decreasing
            # if output.max() - output.min() < 0.2 and target.max() - target.min() > 1:
            #      early_stopping = EarlyStopping(patience=2000, verbose=False, path=model_name)

            if early_stopping.early_stop: # if the validation loss is not decreasing, stop training
                break

        net.load_state_dict(torch.load(model_name)) 
        print(epoch)
        torch.save(net.state_dict(), join(save_path, 'Best_Model' +'_' + str(predict_index[0][:10])))

        net.eval() # set the model to eval mode
        for data_val, target in validloader2: # test set

            output = net(data_val.float().to(device)) # forward pass

        predict = output.float().view(-1).detach().cpu().numpy() # get the predictions
        predict = y_stats.back(predict) # denormalize the predictions, get the real predictions

        predict = np.reshape(predict, (-1, len(namelist)), 'F') # reshape the predictions
        test_y = np.reshape(test_y, (-1, len(namelist)), 'F') # reshape the targets
        plot_valid = np.concatenate((predict, test_y), axis=1) # concatenate the predictions and targets
        plot_valid = pd.DataFrame(plot_valid) # convert to dataframe
        plot_valid.index = date[(np.where(date == predict_index[0])[0][0] + 1):(
                np.where(date == predict_index[-1])[0][0] + 2)] # set the index
        plot_valid.columns = [x + 'out' for x in namelist] + [x + 'real' for x in namelist] # set the column names
        return plot_valid # return the predictions and targets


    def run(self, window_length, train_size, Epoch_num = 2, pre = True):
        T = int(self.a.x.shape[0]/len(namelist))
        result_list = []
        if args.freq != 'daily':
            start_index = np.where(self.a.idx == '2015-06-30'+'/'+'16:00')[0][0]
        else:
            start_index = np.where(self.a.idx == '2015-06-30')[0][0]
        for start in range(start_index,T-1, window_length):
            print(self.a.idx[start])
            if start + window_length <= T - 1: # if the testing period is not the last testing period
                # append the predictions and targets to the result list
                result_list.append(
                    self.train(Epoch_num=Epoch_num, train_index=self.a.idx[start_index - train_size:start],
                               predict_index=self.a.idx[start:start + window_length], lr=None, names=None, pre=pre))
            else: # if the testing period is the last testing period
                result_list.append(
                    self.train(Epoch_num=Epoch_num, train_index=self.a.idx[start_index - train_size:start],
                               predict_index=self.a.idx[start: T - 1], lr=None, names=None, pre=pre))
        return result_list # return the result list


if __name__ == '__main__':
    q = rolling_predict(back_day=args.back_day,
                        lr=0.001)
    result = q.run(args.window_length, args.train_size, Epoch_num=200, pre=False)
    MYDIR = 'hf_' + args.freq + '/dnn'
    CHECK_FOLDER = os.path.isdir(MYDIR)
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
    if args.market == 0:
        pd.concat(result).to_csv(MYDIR + '/meta' + str(args.index) + '.csv')
    elif args.market == 1:
        pd.concat(result).to_csv(MYDIR + '/aug' + str(args.index) + '.csv')

    if args.market == 0:
        if args.freq != 'daily':
            lstm_result = pd.read_csv(MYDIR + '/meta' + str(args.index) + '.csv', index_col=0).loc['2015-07-01/09:30':]
        else:
            lstm_result = pd.read_csv(MYDIR + '/meta' + str(args.index) + '.csv', index_col=0).loc['2015-07-01':]
        report_df = pd.DataFrame(index=namelist, columns=['MSE', 'r2_score'])
        for i in namelist:
            report_df.loc[i, 'MSE'] = mean_squared_error(lstm_result[i + 'out'], lstm_result[i + 'real'])
            report_df.loc[i, 'r2_score'] = r2_score(lstm_result[i + 'real'], lstm_result[i + 'out'])
            report_df.loc[i, 'MAPE'] = mean_absolute_percentage_error(lstm_result[i + 'real'], lstm_result[i + 'out'])
        report_df.to_csv(MYDIR + '/meta_report' + str(args.index) + '.csv')

    elif args.market == 1:
        if args.freq != 'daily':
            lstm_result = pd.read_csv(MYDIR + '/aug' + str(args.index) + '.csv', index_col=0).loc['2015-07-01/09:30':]
        else:
            lstm_result = pd.read_csv(MYDIR + '/aug' + str(args.index) + '.csv', index_col=0).loc['2015-07-01':]
        report_df = pd.DataFrame(index=namelist, columns=['MSE', 'r2_score'])
        for i in namelist:
            report_df.loc[i, 'MSE'] = mean_squared_error(lstm_result[i + 'out'], lstm_result[i + 'real'])
            report_df.loc[i, 'r2_score'] = r2_score(lstm_result[i + 'real'], lstm_result[i + 'out'])
            report_df.loc[i, 'MAPE'] = mean_absolute_percentage_error(lstm_result[i + 'real'], lstm_result[i + 'out'])
        report_df.to_csv(MYDIR + '/aug_report' + str(args.index) + '.csv')