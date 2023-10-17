import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from datetime import datetime
import json
import os
from typing import Dict
from sklearn import linear_model

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
ret_data = pd.read_csv('/Users/beneverman/Documents/Coding/QuantHive/IDVF-Oxford-v1/data/processed-5yr-93-minute/65min_log_ret.csv', index_col=0)

back_day = 6*20
back_day = list(range(back_day))
window_length = 6*60
train_size = 1000 #! MODIFIED for smaller dataset

rv_ppd, date, namelist = initial_preprocess(rv_data) # preprocess data
ret_ppd, date2, namelist2 = initial_preprocess(ret_data) # preprocess data

assert (date == date2).all() # check that dates are the same
assert namelist == namelist2 # check that names are the same

rv_ppd.columns = [col + '_logvol' for col in rv_ppd.columns] # rename columns
ret_ppd.columns = [col + '_ret' for col in ret_ppd.columns] # rename columns

data = pd.concat([rv_ppd, ret_ppd], axis=1) # combine data

# ! No change from linear script 
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

class rolling_predict():
    def __init__(self, keywords = ['XVZ_volatility'], back_day = list(range(0,15)), lr = 0.001):
        self.back_day = back_day
        self.lr = lr
        self.keywords = keywords
        temp = [data[namelist[0]+'_logvol'],data[namelist[0]+'_ret']] # ! This is new, using ret
        self.a = preprocess(temp,data[namelist[0]+'_logvol'], back_day = back_day)

        for ind in namelist[1:]:
            temp = []
            for i in [ ind+'_logvol',ind+'_ret']: # ! This is new, using ret

                temp.append(data[i])
            temp_a = preprocess(temp,data[ind+'_logvol'], back_day = back_day)
            self.a.x = np.concatenate([self.a.x, temp_a.x],axis = 0 )
            self.a.y = np.concatenate([self.a.y, temp_a.y],axis = 0 )
            self.a.idx = np.concatenate([self.a.idx, temp_a.idx],axis = 0 )

    def train(self, train_index, predict_index,  lr,  names, Epoch_num = 300, pre = True):

        clf = linear_model.Lasso(alpha=1e-4,tol=1e-3) # ! new model

        temp_train_start = np.where(self.a.idx == train_index[0])
        temp_index_train = []
        for i in temp_train_start[0]:
            temp_index_train.extend(list(range(i,i+len(train_index))))
        temp_predict_start = np.where(self.a.idx == predict_index[0])
        temp_index_predict= []
        for i in temp_predict_start[0]:
            temp_index_predict.extend(list(range(i,i+len(predict_index))))
        train_x = self.a.x[temp_index_train]
        train_y = self.a.y[temp_index_train]
        test_x = self.a.x[temp_index_predict]
        test_y = self.a.y[temp_index_predict]

        # ! new stuff
        train_x = train_x.reshape(train_x.shape[0], -1)
        test_x = test_x.reshape(test_x.shape[0], -1)

        clf.fit(train_x,train_y)

        predict = clf.predict(test_x)

        # plot_valid_exp = np.exp(plot_valid[['out','real']])
        # print(np.mean((plot_valid_exp.iloc[:,0]-plot_valid_exp.iloc[:,1])**2))
        # plt.plot(train_list)
        # plt.plot(valid_list)
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

        # ! MODIFIED CODE =========================================================
        test_start_date = '2020-06-30' # this is the last date of train, so test will start on the next day
        test_start_date = pd.Timestamp(f'{test_start_date} 10:35', tz='US/Eastern') #! this dataset uses intervals form 10:35-16:00 so this should be 10:35 not 9:30
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
    SAVE_DIR = "/Users/beneverman/Documents/Coding/QuantHive/IDVF-Oxford-v1/outputs/lasso-replication-v1/"
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