import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from datetime import datetime
import json
import os
from typing import Dict, List
from abc import ABC, abstractmethod


def initial_preprocess(data: pd.DataFrame, windsorize=True):
    """
    Handle missing values, convert datetime to index, windsorize outliers, rename columns
    """
    data = data.ffill()
    data = data.bfill()
    assert data.isna().sum().sum() == 0

    data['datetime'] = pd.to_datetime(data['datetime'], utc=True).dt.tz_convert('US/Eastern')
    data.set_index('datetime', inplace=True)

    data.columns = [col for col in data.columns if col != 'datetime'] # filter out datetime
    namelist = data.columns.tolist() # save column names (before renaming)
    data.columns = [col + '_logvol' for col in data.columns] # rename columns
    date = data.index

    if windsorize:
        for clm in data.columns:
            max_p = np.percentile(data[clm], 99.9)
            min_p = np.percentile(data[clm], 0.1)

            data.loc[data[clm] > max_p, clm] = max_p
            data.loc[data[clm] < min_p, clm] = min_p
    
    return data, date, namelist

class Preprocess:
    def __init__(self, input_dfs: List[pd.DataFrame], target, back_day: List, forward_day: int):
        self.input_dfs = input_dfs
        self.target = target
        self.back_day = back_day
        self.forward_day = forward_day

        self.obs_unprocessed = self.input_dfs[0].shape[0] # get the number of observations
        self.eop = self.obs_unprocessed - len(self.back_day) - (forward_day-1) # calculate the expected number of observations after preprocessing 
        
        self._generate_shifted_sequences()
        self._generate_target_mask_and_date()

    def _generate_shifted_sequences(self):
        self.x = []
        for df in self.input_dfs: # for each input df
            shifted_df = pd.concat(
                [df.shift(n) for n in self.back_day], axis=1 # shift and concatenate
            ).reset_index(drop=True).iloc[:, ::-1] # reset index and reverse order

            self.x.append(np.expand_dims(np.array(shifted_df), axis=2)) # expand dims and append on the new dim

        self.x = np.concatenate(tuple(self.x), axis=2) # concatenate on the new dim (now we have a single 3d array)
        # Sanity check - shape should not change until we apply the mask later
        # (num observations, num back days, num features) (7516, 15, 1)
        assert self.x.shape == (self.input_dfs[0].shape[0], len(self.back_day), len(self.input_dfs)),\
            f"Input shape is incorrect, expected {(self.input_dfs[0].shape[0], len(self.back_day), len(self.input_dfs))} but got {self.x.shape}"

    def _generate_target_mask_and_date(self):
        non_na_mask = [~np.any(np.isnan(p)) for p in self.x] # make a mask for non-nan values over all features
        
        # print(f"Target type: {type(self.target)}") #? Debug
        if type(self.target) == pd.Series: # if the target is a series
            self.target = pd.DataFrame(self.target) # convert to df to avoid error in line below where we call .all() and axis=1
        
        self.y = self.target.shift(-self.forward_day).reset_index(drop=True) # shift target
        valid_target_mask = self.y.notna().all(axis=1) # make a mask for valid target values
        
        self.final_mask = np.logical_and(non_na_mask, valid_target_mask) # combine masks
        
        self.x = self.x[self.final_mask] # apply mask
        self.y = np.array(self.y[self.final_mask].reset_index(drop=True)) # apply mask
        self.idx = self.target.index[self.final_mask] # get all dates

        # Sanity check - shape should be reduced by the number of back days
        # (num observations, num features) (7502, 1)
        assert self.x.shape == (self.eop, len(self.back_day), len(self.input_dfs)),\
            f"Input shape is incorrect, should be {self.eop} x {len(self.back_day)}, {len(self.input_dfs)}, is {self.x.shape}"

class Model(ABC):
    @abstractmethod
    def train(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass

class LinearRegression:
    def __init__(self, fit_intercept=True):
        self.coefficients = None
        self.fit_intercept = fit_intercept  # Added

    def train(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError("Mismatched dimensions: X has {} rows but y has {} rows".format(X.shape[0], y.shape[0]))

        if self.fit_intercept:
            X = self._add_bias_term(X)  # Added

        self.coefficients = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        if self.coefficients is None:
            raise ValueError("Model has not been trained yet.")

        if self.fit_intercept: 
            X = self._add_bias_term(X)  # Added

        return X @ self.coefficients

    def _add_bias_term(self, X):  # Added
        return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)  # Added

class RollingPredict:
    def __init__(self, back_day: List, data: pd.DataFrame, namelist: List, window_length: int, forward_day: int = 1):
        self.back_day = back_day
        self.data = data # store data in attribute
        self.namelist = namelist # store namelist in attribute
        self.window_length = window_length
        self.forward_day = forward_day

        # ! Some assertions to make sure everything is good
        self.obs_unprocessed = self.data.shape[0] 
        self.expected_obs_processed = self.obs_unprocessed - len(self.back_day) - (forward_day-1)
        # 7516 - 14 - + 1

        self._rolling_preprocess() # preprocess all the data fort his window

        # some statment to figure out how many models should be trained (based on the window length)
        # would be total number of data points 

        eop = self.expected_obs_processed * len(self.namelist) # expected observations per ticker
        x_len = self.preprocess_obj.x.shape[0]
        y_len = self.preprocess_obj.y.shape[0]
        idx_len = self.preprocess_obj.idx.shape[0]

        assert x_len == eop, f"X shape is incorrect: expected {eop} but got {x_len}"
        assert y_len == eop, f"Y shape is incorrect: expected {eop} but got {y_len}"
        assert idx_len == eop, f"IDX shape is incorrect expected {eop} but got {idx_len}"

    def _rolling_preprocess(self):
        initial_df = [self.data[self.namelist[0]+'_logvol']] # get the first df
        
        self.preprocess_obj = Preprocess(initial_df, self.data[self.namelist[0]+'_logvol'], self.back_day, self.forward_day)

        assert self.preprocess_obj.x.shape[0] == self.expected_obs_processed, "Initial preprocess failed - debug in the preprocess class"

        for ticker in self.namelist[1:]:
            temp_df = [self.data[ticker + '_logvol']]
            temp_preprocess_obj = Preprocess(temp_df, self.data[ticker + '_logvol'], self.back_day, self.forward_day)

            self._concatenate_preprocessed_data(temp_preprocess_obj)

    def _concatenate_preprocessed_data(self, temp_preprocess_obj):
        self.preprocess_obj.x = np.concatenate([self.preprocess_obj.x, temp_preprocess_obj.x], axis=0)
        self.preprocess_obj.y = np.concatenate([self.preprocess_obj.y, temp_preprocess_obj.y], axis=0)
        self.preprocess_obj.idx = np.concatenate([self.preprocess_obj.idx, temp_preprocess_obj.idx], axis=0)

    def train(self, train_index, predict_index):
        # Find the indices that correspond to the starting points of our training window.
        temp_train_start = np.where(self.preprocess_obj.idx == train_index[0])[0]
        
        # Create a list of indices that correspond to all data points in the training window.
        # For each starting point found, we create a range that defines our actual training window.
        temp_index_train = [i for start in temp_train_start for i in range(start, start + len(train_index))]
        
        # same for predict
        temp_predict_start = np.where(self.preprocess_obj.idx == predict_index[0])[0]
        temp_index_predict = [i for start in temp_predict_start for i in range(start, start + len(predict_index))]
        
        train_x = self.preprocess_obj.x[temp_index_train]
        train_y = self.preprocess_obj.y[temp_index_train]
        test_x = self.preprocess_obj.x[temp_index_predict]
        test_y = self.preprocess_obj.y[temp_index_predict]
        
        # reshape features
        train_x = train_x.reshape(train_x.shape[0], -1)
        test_x = test_x.reshape(test_x.shape[0], -1)
        
        model = LinearRegression()
        model.train(train_x, train_y)
        
        predictions = model.predict(test_x) 
        
        # Reshape data for side-by-side comparison.
        predictions = np.reshape(predictions, (-1, len(self.namelist)), 'F')
        test_y = np.reshape(test_y, (-1, len(self.namelist)), 'F')

        results_df = pd.DataFrame(np.concatenate((predictions, test_y), axis=1))
        
        # ! CONCISE VERSION
        # results_df.index = self.preprocess_obj.idx[
        #                    (np.where(self.preprocess_obj.idx == predict_index[0])[0][0] + 1):
        #                    (np.where(self.preprocess_obj.idx == predict_index[-1])[0][0] + 2)]
        # results_df.columns = [x + '_out' for x in self.namelist] + [x + 'real' for x in self.namelist]

        # ! VERSION FOR READABILITY 
        start_idx_position = np.where(self.preprocess_obj.idx == predict_index[0])[0][0] # find the index of the first predict index
        end_idx_position = np.where(self.preprocess_obj.idx == predict_index[-1])[0][0] # find the index of the last predict index

        # Adjust the starting and ending positions to match the DataFrame index.
        adjusted_start = start_idx_position + 1
        adjusted_end = end_idx_position + 2

        results_df.index = self.preprocess_obj.idx[adjusted_start:adjusted_end]

        predicted_columns = [x + 'out' for x in self.namelist]
        actual_columns = [x + 'real' for x in self.namelist]
        results_df.columns = predicted_columns + actual_columns

        # Step 13: Return the results DataFrame.
        return results_df

    def run(self, window_length, train_size, test_start_dt_str: str, Epoch_num=0, lr=None,):
        # Calculate the total number of observations
        T = int(self.preprocess_obj.x.shape[0] / len(self.namelist))
        result_list = []

        # Set the starting date for the test data
        test_start_dt = pd.Timestamp(test_start_dt_str, tz='US/Eastern')

        assert test_start_dt in self.preprocess_obj.idx, "Test start date is not in the index"
        start_index = np.where(self.preprocess_obj.idx == test_start_dt)[0][0] 

        num_windows = int((T - train_size) / window_length) + 1 # ? make sure this work 

        for start in range(start_index, T - 1, window_length):
            print(self.preprocess_obj.idx[start]) #? Debugging

            # Determine the range for training and prediction
            train_idx_start = self.preprocess_obj.idx[start_index - train_size:start]
            
            # min function to handle the edge case for the last window
            predict_idx_range = self.preprocess_obj.idx[start:min(start + window_length, T-1)]

            kwargs = {
                "train_index": train_idx_start,
                "predict_index": predict_idx_range,
            }

            if Epoch_num != 0: kwargs["Epoch_num"] = Epoch_num # for the NN
            if lr != None: kwargs["lr"] = lr # for the NN

            result = self.train(**kwargs)
            result_list.append(result)
        # ! ============================================================

        return result_list

def _make_report(result: pd.DataFrame, namelist):
        report_df = pd.DataFrame(index = namelist,columns=['MSE','r2_score']) # init report df with MSE and r square
        
        test_start_date = '2020-06-30' # this is the last date of train, so test will start on the next day
        test_start_date = pd.Timestamp(f'{test_start_date} 10:35', tz='US/Eastern') #! this dataset uses intervals form 10:35-16:00 so this should be 10:35 not 9:30
        # start the result df freom the test start date
        result = result.loc[test_start_date:]

        for i in namelist:
            # report_df.loc[i,'MSE'] = mean_squared_error(result[i+'out'],result[i+'real']) # calculate MSE
            # ! this original code is technically backwards but its squared so it doesn't matter
            report_df.loc[i,'MSE'] = mean_squared_error( result[i + 'real'],result[i + 'out']) # calculate MSE
            report_df.loc[i,'r2_score'] = r2_score( result[i + 'real'],result[i + 'out']) # calculate r square
            report_df.loc[i,'MAPE'] = mean_absolute_percentage_error( result[i + 'real'],result[i + 'out']) # calculate MAPE
        return report_df


def _save(result: pd.DataFrame, report_df: pd.DataFrame, args_dict: Dict = None):
    SAVE_DIR = "/Users/beneverman/Documents/Coding/QuantHive/IDVF-Oxford-v1/outputs/linear-refactored-v1/"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    subdir = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(os.path.join(SAVE_DIR, subdir))

    result.to_csv(os.path.join(SAVE_DIR, subdir, "result.csv")) # save the result df
    report_df.to_csv(os.path.join(SAVE_DIR, subdir, "report.csv")) # save the report

    with open(os.path.join(SAVE_DIR, subdir, "args.json"), "w") as f: # save the args
        json.dump(args_dict, f, indent=4)


def main():

    back_day = 15
    back_day = list(range(back_day))
    window_length = 6*250
    train_size = 1000 #! MODIFIED for smaller dataset
    forward_day = 1

    rv_data = pd.read_csv('/Users/beneverman/Documents/Coding/QuantHive/IDVF-Oxford-v1/data/processed-5yr-93-minute/65min_rv.csv', index_col=0)
    rv_data, date, namelist = initial_preprocess(rv_data, windsorize=False)

    test_start_dt_str = '2020-06-30 10:35'


    rp_args = [back_day, rv_data, namelist, window_length, forward_day]
    q = RollingPredict(*rp_args)
    result = q.run(window_length, train_size, test_start_dt_str)
    result = pd.concat(result) # concat the result list into a df

    report_df = _make_report(result, namelist) # make the report df
    _save(result, report_df)


if __name__ == '__main__':
    main()