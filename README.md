## Data
### processed-5yr-93-minute
This includes 5yrs of raw minute data from Polygon from 2018-10-11 to 2023-10-09
The data is processed in 3 ways:
1. 65min aggregates (open, high, low, close, volume)
   1. includes a 9:30 period, but not a 16:00 period
2. 65min Realized Volatility (RV)
   1. includes a 16:00 period instead of a 9:30 period
   2. `preprocess-5y-93-v2.ipynb`
3. 65min log returns
   1. calculated as log((last trade in 65min bin) - log(first trade in 65min bin))
   2. `preprocess-5y-93-v4.ipynb`
   

## Completed
- [X] make a bool to filter out premarket data
   [X] write a func to get a master set of X_min length intervals between tuples of market open/close given by `get_nyse_date_tups`
- [X] use that func to get aggs of all stocks 
  - [X] need to figure out how to handle missing and inconsistent data
- [X] use the agg data to replicate the linear script in `linear-replication-full.ipynb`
  - [X] write unit tests/sanity checks for preprocessing class
  - [X] Fix rolling predict method, add tests/sanity checks
- [X] sanity check the linear script
  - [X] do some out of sample testing if not already done in the rolling predict (go back and check the paper)
- [X] get Lasso running

## TODO

- [ ] get HARD running
  - [ ] diff check it against LASSO (if it uses ret) or Linear if it doesn't
  - [ ] get it working in a notebook
  - [ ] port over to a script
- [ ] get the MLP running
- [ ] get an LSTM running
- [ ] start to refactor parts of the codebase into reusable modules/rework abstractions
- [ ] run a naive report on new rv dataset