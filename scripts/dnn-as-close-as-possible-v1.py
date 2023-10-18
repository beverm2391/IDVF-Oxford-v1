import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from datetime import datetime
import json
import os
from typing import Dict

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import torch


def initial_preprocess(data, windsorize=True):
    data = data.ffill()
    data = data.bfill()
    assert data.isna().sum().sum() == 0

    data['datetime'] = pd.to_datetime(data['datetime'], utc=True).dt.tz_convert('US/Eastern')
    data.set_index('datetime', inplace=True)

    data.columns = [col for col in data.columns if col != 'datetime'] # filter out datetime
    date = data.index

    namelist = data.columns.tolist()

    if windsorize:
        for clm in data.columns:
            max_p = np.percentile(data[clm], 99.9)
            min_p = np.percentile(data[clm], 0.1)

            data.loc[data[clm] > max_p, clm] = max_p
            data.loc[data[clm] < min_p, clm] = min_p

    return data, date, namelist

rv_data = pd.read_csv('/Users/beneverman/Documents/Coding/QuantHive/IDVF-Oxford-v1/data/processed-5yr-93-minute/65min_rv.csv', index_col=0)

back_day = 6*20
back_day = list(range(back_day))
window_length = 6*250
train_size = 1000 #! MODIFIED for smaller dataset
count_one_day = 6

data, date, namelist = initial_preprocess(rv_data, windsorize=False) # preprocess data, no windsorization

data.columns = [col + '_logvol' for col in data.columns] # rename columns

models_dir = '/Users/beneverman/Documents/Coding/QuantHive/IDVF-Oxford-v1/models/'

if not os.path.exists(models_dir + "temp/"): os.makedirs(models_dir + "temp/") # create models and temp folders if they don't exist

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DNN, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

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


class mean_var():
    def __init__(self, data):
        self.ave = np.mean(data, axis = 0)
        self.var = np.var(data,axis = 0)
    def preprocess(self, temp):
        return (temp - self.ave)/np.sqrt(self.var)
    def back(self, temp):
        return temp * np.sqrt(self.var)+self.ave


class rolling_predict():
    def __init__(self, keywords = ['XVZ_volatility'], back_day = list(range(0,15)), lr = 0.001):
        self.back_day = back_day
        self.lr = lr
        self.keywords = keywords
        temp = [data[namelist[0] + '_logvol']]
        print("Preprocessing data...")
        self.a = preprocess(temp, data[namelist[0] + '_logvol'], back_day=back_day)

        for ind in namelist[1:]:
            temp = []
            for i in [ind + '_logvol']:
                temp.append(data[i])
            temp_a = preprocess(temp, data[ind + '_logvol'], back_day=back_day)
            self.a.x = np.concatenate([self.a.x, temp_a.x], axis=0)
            self.a.y = np.concatenate([self.a.y, temp_a.y], axis=0)
            self.a.idx = np.concatenate([self.a.idx, temp_a.idx], axis=0)

    def train(self, train_index, predict_index,  lr,  names, Epoch_num = 300, pre = True):
        print(f"Training on {train_index[0]} - {train_index[-1]}")
        print(f"Predicting on {predict_index[0]} - {predict_index[-1]}")

        val_set_size = 100
        assert val_set_size < window_length / count_one_day,\
            f"val_set_size {val_set_size} must be less than {window_length / count_one_day} (window_length / count_one_day)"

        temp_train_start = np.where(self.a.idx == train_index[0])
        assert temp_train_start != [], f"train_index[0] {train_index[0]} not in index {self.a.idx}"

        # training set
        temp_index_train = []
        for i in temp_train_start[0]:
            temp_index_train.extend(list(range(i, i + len(train_index) - val_set_size * count_one_day))) #! offset by val_set_size days - i think this is to create the validation set

        # validation set
        temp_index_valid = []
        for i in temp_train_start[0]:
            temp_index_valid.extend(list(range(i + len(train_index) - val_set_size * count_one_day, i + len(train_index))))

        # testing set
        temp_predict_start = np.where(self.a.idx == predict_index[0])
        temp_index_predict = []
        for i in temp_predict_start[0]:
            temp_index_predict.extend(list(range(i, i + len(predict_index))))

        # make datasets
        train_x = self.a.x[temp_index_train]
        train_y = self.a.y[temp_index_train]
        valid_x = self.a.x[temp_index_valid]
        valid_y = self.a.y[temp_index_valid]
        test_x = self.a.x[temp_index_predict]
        test_y = self.a.y[temp_index_predict]

        x_stats = mean_var(train_x) #! normalize
        y_stats = mean_var(train_y) #! normalize

        train_x = x_stats.preprocess(train_x)
        train_y = y_stats.preprocess(train_y)

        valid_x = x_stats.preprocess(valid_x)
        valid_y = y_stats.preprocess(valid_y)

        test_x  = x_stats.preprocess(test_x)
        #test_y  = y_stats.preprocess(test_y)

        train_x = train_x.reshape(train_x.shape[0], -1)
        test_x = test_x.reshape(test_x.shape[0], -1)
        valid_x = valid_x.reshape(valid_x.shape[0], -1)

        from lib.earlystopping import EarlyStopping #! early stopping

        if lr is None: lr = self.lr # if lr not passed to train, use default lr for RP class
        if names is None: names = self.keywords # if names not passed to train, use default names for RP class

        trainloader = DataLoader(TensorDataset(torch.tensor(train_x), torch.tensor(train_y)),1024,
                                 shuffle=True) #! batch size 1024, shuffle=True, TensorDataset loafs all data into memory
        validloader1 = DataLoader(TensorDataset(torch.tensor(valid_x), torch.tensor(valid_y)),
                                  torch.tensor(valid_x).shape[0],
                                  shuffle=False) #! batch size = valid_x.shape[0] (num in val set, 250), shuffle=False
        validloader2 = DataLoader( TensorDataset(torch.tensor(test_x), torch.tensor(test_y)),
            len(torch.tensor(test_y)), shuffle=False) #! batch size = len(test_y), shuffle=False

        early_stopping = EarlyStopping(patience=10, verbose=False, path=f"{models_dir}temp/checkpoint.pt") # early stopping
        net = DNN(train_x.shape[1], 1).to(device) # init the dnn, move it to gpu if available
        loss_function = nn.MSELoss() # loss function
        optimiser = optim.Adam(net.parameters(), lr=lr, eps=1e-8) # optimizer
        Epoch_num = Epoch_num # not sure why this is here

        for epoch in range(Epoch_num + 1):
            net.train() # set model to train mode
            for data_val, target in trainloader: # iterate over batches
                optimiser.zero_grad() # zero gradients
                output = net(data_val.float().to(device)) # forward pass
                loss = loss_function(output.float().view(-1), target.float().view(-1).to(device)) # compute loss
                loss.backward() # backprop
                optimiser.step() # update weights
            net.eval() # set model to eval mode
            for data_val, target in validloader1: 
                output = net(data_val.float().to(device)) # forward pass
                loss_valid = loss_function(output.float().view(-1), target.float().view(-1).to(device)) # compute loss

            # valid_list.append(loss_valid.float().view(-1).detach().cpu().numpy()[0])
            # test_list.append(loss_test.float().view(-1).detach().cpu().numpy()[0])
            early_stopping(loss_valid.detach().cpu().numpy().reshape(-1)[0], net) # early stopping
            # if output.max() - output.min() < 0.2 and target.max() - target.min() > 1:
            #      early_stopping = EarlyStopping(patience=2000, verbose=False, path=model_name)

            tl = loss_valid.float().view(-1).detach().cpu().numpy()[0]
            vl = loss_valid.float().view(-1).detach().cpu().numpy()[0]

            print(f"Epoch: {epoch} of {Epoch_num} | Train Loss: {tl:0.4f} | Valid Loss: {vl:0.4f}")

            if early_stopping.early_stop:
                break

        net.load_state_dict(torch.load(f"{models_dir}temp/checkpoint.pt")) # load best model
        # print(epoch)
        model_name = f'Best_Model_{str(predict_index[0])}-{str(predict_index[-1])}.pt'
        torch.save(net.state_dict(), os.path.join(models_dir, model_name)) # save best model

        net.eval() # set model to eval mode
        for data_val, target in validloader2:

            output = net(data_val.float().to(device)) # forward pass

        predict = output.float().view(-1).detach().cpu().numpy() # get predictions
        predict = y_stats.back(predict) # denormalize predictions

        predict = np.reshape(predict, (-1, len(namelist)), 'F') # reshape predictions, fortran order
        test_y = np.reshape(test_y, (-1, len(namelist)), 'F') # reshape test_y, fortran order
        plot_valid = np.concatenate((predict, test_y), axis=1) 
        plot_valid = pd.DataFrame(plot_valid)
        plot_valid.index = date[(np.where(date == predict_index[0])[0][0] + 1):(
                np.where(date == predict_index[-1])[0][0] + 2)]
        plot_valid.columns = [x + 'out' for x in namelist] + [x + 'real' for x in namelist]
        return plot_valid
    
    def run(self, window_length, train_size, Epoch_num = 2, pre = True):
        T = int(self.a.x.shape[0]/len(namelist))
        result_list = []

        # ! MODIFIED CODE =========================================================
        test_start_date = '2020-06-30' # this is the last date of train, so test will start on the next day
        test_start_date = pd.Timestamp(f'{test_start_date} 10:35', tz='US/Eastern') #! this dataset uses intervals form 10:35-16:00 so this should be 10:35 not 9:30
        assert test_start_date in self.a.idx, f"test_start_date {test_start_date} not in index"
        start_index = np.where(self.a.idx == test_start_date)[0][0]
        # ! END MODIFIED CODE =====================================================

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


q = rolling_predict( back_day= back_day,
            lr=0.001)
result = q.run(window_length, train_size, Epoch_num = 2,pre = False)
# ! END UNMODIFIED CODE =========================================================

# ! REPORTING AND SAVING (modified) ================================================
def _make_report(result: pd.DataFrame):
        report_df = pd.DataFrame(index = namelist,columns=['MSE','r2_score']) # init report df with MSE and r square
        
        test_start_date = '2020-06-30' # this is the last date of train, so test will start on the next day
        test_start_date = pd.Timestamp(f'{test_start_date} 10:35', tz='US/Eastern') #! this dataset uses intervals form 10:35-16:00 so this should be 10:35 not 9:30
        # start the result df freom the test start date
        result = result.loc[test_start_date:]

        for i in namelist:
            # report_df.loc[i,'MSE'] = mean_squared_error(result[i+'out'],result[i+'real']) # calculate MSE
            # ! this original code is technically backwards but its sqared so it doesn't matter
            report_df.loc[i,'MSE'] = mean_squared_error( result[i + 'real'],result[i + 'out']) # calculate MSE
            report_df.loc[i,'r2_score'] = r2_score( result[i + 'real'],result[i + 'out']) # calculate r square
            report_df.loc[i,'MAPE'] = mean_absolute_percentage_error( result[i + 'real'],result[i + 'out']) # calculate MAPE
        return report_df
    
def _save(result: pd.DataFrame, report_df: pd.DataFrame, args_dict: Dict = None):
    SAVE_DIR = "/Users/beneverman/Documents/Coding/QuantHive/IDVF-Oxford-v1/outputs/dnn-replication-v1/"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    subdir = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(os.path.join(SAVE_DIR, subdir))

    result.to_csv(os.path.join(SAVE_DIR, subdir, "result.csv")) # save the result df
    report_df.to_csv(os.path.join(SAVE_DIR, subdir, "report.csv")) # save the report

    with open(os.path.join(SAVE_DIR, subdir, "args.json"), "w") as f: # save the args
        json.dump(args_dict, f, indent=4)

result = pd.concat(result) # concat the result list into a df
report_df = _make_report(result) # make the report df

_save(result, report_df)