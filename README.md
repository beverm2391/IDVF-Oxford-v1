# Intraday Volatility Forecasting with Deep Learning


## Background
This repo serves to mark my efforts to forecast volatility using deep learning. While working on proprietary deep learning models to forecast volatility, I found [this]([/Users/beneverman/Documents/Coding/QuantHive/IDVF-Oxford-v1/IDVF_Oxford.pdf](https://academic.oup.com/jfec/advance-article/doi/10.1093/jjfinec/nbad005/7081291)) paper by a group of researchers as Oxford. The paper was fascinating, as the researchers discovered commonality between securities in their intraday volatility patterns. They leveraged this commonality to train forecasting models on a subset of securities. 

Academic research serves to introduce novel ideas and methodologies, typical in the absence of a specific application or use. My interest in forecasting volatility is driven by the fact that, in theory, one who can accurately forecast volatility can leverage options strategies to generate significant returns.

## Replication

The first author, [Chao Zhang](https://sites.google.com/view/chaozhang94/) was kind enough to share the original code, which served as a starting point for this replication - which now includes a custom dataset and fully refactored code for readability and reusability. The original dataset, [Lobster](https://lobsterdata.com/) was more expensive than we liked, so I wrote a custom client to pull large amounts of minute data from [Polygon](https://polygon.io/). That was a project in and of itself, can can be found in [this repo](https://github.com/beverm2391/AsyncFetcher-v1). Then, I reverse engineered the original code to understand the data processing and model training. I refactored the code to be more modular and extensible. 

I was able to produce similar results to the original paper using my new dataset and refactored code. The `dnn-refactored-v1.py` script is the most performant model (MLP).

## Data

*I have not included my dataset in this repo, as it is too large. If you want it, shoot me an [email](mailto:evermanben@gmail.com) and I'll send it to you.*

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
- [X] get HARD running
  - [X] diff check it against LASSO (if it uses ret) or Linear if it doesn't
  - [X] get it working in a notebook
  - [X] port over to a script
- [X] get the MLP running
- [X] start to refactor parts of the codebase into reusable modules/rework abstractions
  - [X] figure out why the run() func is only looping 3 times (should be 4?) - add some assert statement in for expected vs actual num models

## TODO

- [ ] add the LASSO and HARD model classes
- [ ] add the MLP model class
- [ ] implement the training loop for the MLP
- [ ] run a naive report on new rv dataset
- [ ] start comparing the different models by changing 1 thing at a time and observing the difference
- [ ] get an LSTM running
