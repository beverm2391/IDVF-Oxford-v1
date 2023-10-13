## Completed
- [X] make a bool to filter out premarket data
   [X] write a func to get a master set of X_min length intervals between tuples of market open/close given by `get_nyse_date_tups`
- [X] use that func to get aggs of all stocks 
  - [X] need to figure out how to handle missing and inconsistent data

## TODO
- [ ] use the agg data to replicate the linear script in `linear-replication-full.ipynb`
  - [X] write unit tests/sanity checks for preprocessing class
  - [X] Fix rolling predict method, add tests/sanity checks
  - [ ] Make sure the Epoch_num passed to the RollingPredict.run() method is actually used
- [ ] replicate the Lasso script