from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_percentage_error
from time import perf_counter
from dataclasses import dataclass, asdict
import os
import json

from lib.utils import get_65min_aggs, rv

def main(main_args : MainArguments):
    # ! Unpack args ===========================================================

    back_day = main_args['back_day']
    forward_day = main_args['forward_day']
    window_length = main_args['window_length']
    SAVE_DIR = main_args['save_dir']


    # ! DATA PREPARATION =======================================================

    data = get_65min_aggs() # get the 65 minute aggregate bars
    num_agg_bars = data.shape[0] # number of aggregate bars

    data.ffill(inplace=True)
    data.bfill(inplace=True)
    assert data.isna().sum().sum() == 0

    data['datetime'] = pd.to_datetime(data['datetime'], utc=True).dt.tz_convert('US/Eastern')
    data.set_index('datetime', inplace=True)

    namelist = data.columns.tolist()

    for ind in namelist:
        data[ind + "_logvol"] = rv(data[ind], window_length)

    date = data.index

    assert data.shape == (num_agg_bars, len(namelist)*2), "Dataframe shape is incorrect, should be the same number of rows as raw data but with double columns because we should have a price col and vol col for each ticker "

    # ! PREPROCESSING ===========================================================

    class Preprocess:
        # ! They put all the logic in the class constructor - i would prefer to have a separate function for each step
        def __init__(self, input : List[pd.DataFrame], target, back_day: list = back_day, forward_day = forward_day):
            # this liss(range(0,15)), forward_day = 1 seem to correspond to the lookback window and the forecast horizon
            # ! input is a list of dataframes, for example [price,volatility] with index as the same as target.
            # ! note that the original code in rolling_predict passes one stock at a time, so the input is a list of one dataframe
            # list of dfs holds all the results, target is the actual "input" column
            
            # ! Section 1 - make incrementally shifted seqences for each df in the input list
            self.x = []
            for df in input:
                # Shift the dataframe by each value in back_day and concatenate along columns
                # ! detailed explanation below
                shifted_df = pd.concat(
                    list(map(lambda n: df.shift(n), back_day)), axis=1
                ).reset_index(drop=True).loc[:, ::-1]
                self.x.append(np.expand_dims(np.array(shifted_df), axis=2)) # Expand dimensions to make it compatible for future concatenation

            self.x = np.concatenate(tuple(self.x), axis=2) # Concatenate all processed input data along the last axis to return a 3D array

            # ! X shape = (7516, 15, 1), rows / columns / channels
            # ! X shape = (number of agg bars) / (back day list len) / (#dfs in input list)
            assert self.x.shape == (input[0].shape[0], len(back_day), len(input)), "Input shape is incorrect" # ! this is a sanity check to make sure the shape is correct
            
            # ! Section 2 - make the target, mask, and date
            idx1 = [~np.any(np.isnan(p)) for p in self.x] # Create an index mask where none of the elements in the x dataframes are NaN
            # ! for each row in x (which includes the row data from all dfs), if any of the values are NaN, then return False, else return True (therefore the length of idx will be the number of rows)
            self.y = target.shift(-forward_day) # Shift the target by forward_day to align with predictor variables
            self.y = pd.DataFrame(self.y).reset_index(drop=True) # Reset index to align with self.x (i removed the double parentheses around self.y)
            self.idx2 = self.y.notna().all(axis=1) # simple mask all rows where there are no NaNs (i.e. all values are present) note that this is notna, not isna like before. So this is the opposite of the previous mask
            self.idx = np.logical_and(idx1, self.idx2) # Combine the two index masks (element-wise and)
            # ! final mask "idx" is of shape (rows, )

            self.x = self.x[self.idx] # Filter x and y data based on combined index mask
            self.y = np.array(self.y[self.idx].reset_index(drop=True)) # Filter date based on combined index mask, make it an array (this changes expected dims)
            
            # ! Section 3 - make the date
            self.idx = data.index[self.idx] #! this is weird naming convention, because now self.idx is the date index from the data df, not a mask anymore

    # ! MODEL ===================================================================
    
    class RollingPredict:
        def __init__(self, back_day: list = back_day):
            self.back_day = back_day # list of days to look back

            first_df = [data[namelist[0]+'_logvol']] # temporary list of dataframes (only one element for now)
            self.a = Preprocess(first_df, data[namelist[0]+'_logvol'], back_day = back_day) # preprocess the data
            # ! self.a is an object with attributes x, y, and idx
            # self.a is a preprocess object
            # ! nans = number of original agg bars - window_length - back_day list length
                # There might be an edge case where the log_vol window length is longer than the window length, which could mess this up
            # ! preprocessed agg bars (ppd agg bars) = number of original agg bars - window_length - back_day list length
            # self.a.x is a numpy array of shape (6001, 15, 1) / (ppd agg bars, back day list len, #dfs in input list)
            # self.a.y is a numpy array of shape (6001, 1) / (ppd agg bars - nans, 1)
            # self.a.idx is a pandas index of timestamps of shape (6001, ) / (ppd agg bars, )

            for ind in namelist[1:]: # iterate over each ticker
                temp = [data[ind + '_logvol']] # make a list of dataframes (only one element for now) (this is passed to the input attribute of Preprocess)
                # Shape of temp = (7516, 1)
                temp_a = Preprocess(temp, data[ind + '_logvol'], back_day = back_day) 

                # ! Expected Dims for temp_a
                # temp_a.x = (6001, 15, 1) / (ppd agg bars, back day list len, #dfs in input list)

                # ! these lines essentially concat onto the first_df object stored in self.a
                # ! So theyre updating the x, y, and idx attributes of self.a
                # ! not sure why they did this
                self.a.x = np.concatenate([self.a.x, temp_a.x], axis=0) # concatenate the x data (predictor variables)
                # now (first loop) self.a.x should be of shape (12002, 15, 1) / (ppd agg bars, back day list len, #dfs in input list)
                self.a.y = np.concatenate([self.a.y, temp_a.y], axis=0) # concatenate the y data (target variable)
                # now (first loop) self.a.y should be of shape (12002, 1) / (ppd agg bars, 1)
                self.a.idx = np.concatenate([self.a.idx, temp_a.idx], axis=0) # concatenate the date data
                # now (first loop) self.a.idx should be of shape (12002, ) / (ppd agg bars, )

            ppd_agg_bars = first_df[0].shape[0] - window_length - len(back_day) # number of agg bars - window length - back day list length
            all_ppd_agg_bars = ppd_agg_bars * len(namelist) # total number of agg bars * number of tickers
            assert self.a.x.shape == (all_ppd_agg_bars, len(back_day), 1), "Preprocessed df should have dimensions (ppd agg bars * name list len, len back day list, 1)"
            assert self.a.y.shape == (all_ppd_agg_bars, 1), "Preprocessed df should have dimensions (ppd agg bars, 1)"
            assert self.a.idx.shape == (all_ppd_agg_bars, ), "Preprocessed df should have dimensions (ppd agg bars, )"

        def train(self, train_index, predict_index):
            assert len(train_index) > 0, "Train index must not be empty"
            assert len(predict_index) > 0, "Predict index must not be empty"

            # ! Need to figure out how all this works ========================
            temp_train_start = np.where(self.a.idx == train_index[0]) # match the date index stored in self.a.idx to the first training date
            # TODO - figure out the data structure of temp_train_start
            temp_index_train = [] # list of indices for training data

            for i in temp_train_start[0]: # for each index in temp_train_start[0]
                temp_index_train.extend(list(range(i, i + len(train_index)))) # add the indices for the training data

            temp_predict_start = np.where(self.a.idx == predict_index[0]) # get the index of the first prediction date
            temp_index_predict = []

            for i in temp_predict_start[0]:
                temp_index_predict.extend(list(range(i, i + len(predict_index)))) # add the indices for the prediction data
            
            train_x = self.a.x[temp_index_train] # get the training predictor data
            train_y = self.a.y[temp_index_train] # get the training target data
            test_x = self.a.x[temp_index_predict] # get the test predictor data
            test_y = self.a.y[temp_index_predict] # get the test target data

            train_x = train_x.reshape(train_x.shape[0], -1) # reshape the training predictor data
            test_x = test_x.reshape(test_x.shape[0], -1) # reshape the test predictor data

            # ! bias term
            train_x = np.concatenate((np.ones((train_x.shape[0], 1)), train_x), axis=1) # add a column of ones to the training predictor data
            test_x = np.concatenate((np.ones((test_x.shape[0], 1)), test_x), axis=1) # add a column of ones to the test predictor data

            # ! get beta
            def _get_beta(x, y):
                """
                Get the coefficients of the linear regression model
                $$\beta = \left( X^T X \right)^{-1} X^T y$$
                """
                return np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T, x)), x.T), y)
            
            beta = _get_beta(train_x, train_y) # get the beta coefficients
            pred = np.matmul(test_x, beta) # get the predictions

            pred = np.reshape(pred, (-1, len(namelist)), 'F') # reshape the predictions, fortan style
            test_y = np.reshape(test_y, (-1, len(namelist)), 'F')

            plot_valid = np.concatenate((pred, test_y), axis=1)
            plot_valid = pd.DataFrame(plot_valid)
            # Locate the indices in the 'date' array where the prediction interval starts and ends
            start_date_index = np.where(date == predict_index[0])[0][0]
            end_date_index = np.where(date == predict_index[-1])[0][0]
            start_date_index += 1 # Increment indices to align with the desired date range
            end_date_index += 2
            plot_valid.index = date[start_date_index:end_date_index] # Create the new date range for the index of 'plot_valid'

            plot_valid.columns = [x + 'out' for x in namelist] + [x + 'real' for x in namelist]
            return plot_valid # returns the prediction and the real value

            # ! ===============================================================
            
        def run(self, window_length, train_size=None):
            # from the paper:
            # Our testing period starts from July 1, 2015 until June 30, 2016, and the corresponding training and validation samples are [July 1, 2011, June 30, 2014] and [July 1, 2014, June 30, 2015],
            test_start_date = '2020-06-30' # this is the last date of train, so test will start on the next day
            test_start_date = pd.Timestamp(f'{test_start_date} 9:30', tz='US/Eastern')

            seq_per_stock = int(self.a.x.shape[0]/len(namelist)) # use int to prevent float. T is the number of agg bars per ticker, so ppd agg bars
            result_list = []
            start_index = np.where(self.a.idx == test_start_date)[0][0] # get the index of the first prediction date for the first ticker

            if train_size is None:
                train_size = start_index # if train size is not specified, then set it to the start index
            if train_size > start_index:
                raise ValueError(f"Train size must be less than the start index. Your train size {train_size} is greater than the start index {start_index}")

            range_len = (seq_per_stock - start_index) // window_length + 1 # Example (expected number of windows)
            for window_start in range(start_index,  seq_per_stock-1, window_length): # creates a range with (start_idx, ppd_aggs -1, window_length) (i, o, step)
                # print(self.a.idx[window_start])

                window_end = window_start + window_length
                training_start = start_index - train_size
                assert training_start >= 0, f"Training start must be greater than 0. Your training size {train_size} is greater than window size {window_length}"
                training_end = window_start # this will increment by window_length each iteration, increasing the training set size by window_length each time

                # ! The if/else logic prevents an index out-of-bounds error that could occur during the last iteration of the rolling window approach, especially when the remaining data points at the end of the dataset are fewer than the specified window_length
                if window_end <= seq_per_stock - 1: # if the end of the window is less than the end of the test set
                    train_indices = self.a.idx[training_start:training_end]
                    predict_indices = self.a.idx[window_start:window_end]
                else:
                    train_indices = self.a.idx[training_start:training_end]
                    predict_indices = self.a.idx[window_start:seq_per_stock - 1]

                result = self.train(
                    train_index=train_indices,
                    predict_index=predict_indices,
                )
                result_list.append(result)
            return result_list
        
    # ! Some definitions ========================================================

    def _make_report(result: pd.DataFrame):
        report_df = pd.DataFrame(index = namelist,columns=['MSE','r2_score']) # init report df with MSE and r square
        for i in namelist:
            report_df.loc[i,'MSE'] = mean_squared_error(result[i+'out'],result[i+'real']) # calculate MSE
            report_df.loc[i,'r2_score'] = r2_score( result[i + 'real'],result[i + 'out']) # calculate r square
            report_df.loc[i,'MAPE'] = mean_absolute_percentage_error( result[i + 'real'],result[i + 'out']) # calculate MAPE
        return report_df
    
    def _save(result: pd.DataFrame, report_df: pd.DataFrame, args_dict: Dict):
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        subdir = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(os.path.join(SAVE_DIR, subdir))

        result.to_csv(os.path.join(SAVE_DIR, subdir, "result.csv")) # save the result df
        report_df.to_csv(os.path.join(SAVE_DIR, subdir, "report.csv")) # save the report
        

        with open(os.path.join(SAVE_DIR, subdir, "args.json"), "w") as f: # save the args
            json.dump(main_args, f, indent=4)
        
    ## ! RUN AND SAVE =====================================================================
    
    q = RollingPredict(back_day=back_day)
    results = q.run(window_length) # ! I dont think this epoch number is changing anything, fix
    result = pd.concat(results)
    report = _make_report(result)
    args_dict = asdict(main_args) # convert the args to a dict

    _save(result, report, args_dict)

# ! MAIN ========================================================================

if __name__ == "__main__": 

    @dataclass # i think this needs to stay outside of if __name__ == "__main__"
    class MainArguments:
        back_day : list
        forward_day : int
        window_length : int
        save_dir : str

    main_args = MainArguments({
        "back_day" : list(range(0, 15)),
        "forward_day" : 1,
        "window_length" : 6*250,
        "save_dir" : '/Users/beneverman/Documents/Coding/QuantHive/IDVF-Oxford-v1/outputs/'
    })

    print("Running linear-replication-v1.py")
    start = perf_counter()
    main(main_args)
    elapsed = perf_counter() - start
    print(f"Finished in {elapsed:0.2f} seconds")