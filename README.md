## Completed
- [X] make a bool to filter out premarket data
   [X] write a func to get a master set of X_min length intervals between tuples of market open/close given by `get_nyse_date_tups`
- [X] use that func to get aggs of all stocks 
  - [X] need to figure out how to handle missing and inconsistent data
- [X] use the agg data to replicate the linear script in `linear-replication-full.ipynb`
  - [X] write unit tests/sanity checks for preprocessing class
  - [X] Fix rolling predict method, add tests/sanity checks

## TODO
- [ ] sanity check the linear script
  - [ ] do some out of sample testing if not already done in the rolling predict (go back and check the paper)
- [ ] replicate the Lasso script