from lib.core import *

# TODO: make this use DNN - right now this is just placeholder

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

    report_df = make_report(result, namelist, test_start_dt_str) # make the report df
    save(result, report_df)

if __name__ == '__main__':
    main()