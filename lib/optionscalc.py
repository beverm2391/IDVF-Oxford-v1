import py_vollib_vectorized
from py_vollib_vectorized import vectorized_black, vectorized_implied_volatility, \
    vectorized_delta, vectorized_vega, vectorized_theta, vectorized_rho, vectorized_gamma, get_all_greeks, price_dataframe
import pandas as pd

# make sure to import this after py_vollib_vectorized
from py_vollib.black import black

# ! i was using this to fix the import problem, but it's not needed anymore due to the import order above
# def ensure_patch(call=0, max_calls=3):
#     print(f"Ensure patch call: {call}")
#     from py_vollib.black import black
#     blk = black
#     if call > max_calls:
#         raise Exception("py_vollib_vectorized is not being used")
#     if type(blk) == py_vollib_vectorized.repr_partial:
#         return blk
#     else:
#         ensure_patch(call=call+1, max_calls=max_calls)

assert type(black) == py_vollib_vectorized.repr_partial, "py_vollib_vectorized is not being used"
# print(black)

class OptionsCalc:
    def __init__(self):
        pass

    def black_scholes(self, flag, S, K, t, r, sigma, return_as="numpy"):
        """
        flag: 'c' or 'p' (call or put)
        S: stock price
        K: strike price
        t: time to expiration (in years)
        r: risk-free interest rate (usually 0.02)
        sigma: volatility of the underlying asset
        return_as: 'numpy' or 'series' or 'dataframe'
        """
        return vectorized_black(flag, S, K, t, r, sigma, return_as=return_as)

    def implied_volatility(self, flag, S, K, t, r, q, model='black_scholes', on_error='ignore', return_as="numpy"):
        """
        flag: 'c' or 'p' (call or put)
        S: stock price
        K: strike price
        t: time to expiration (in years)
        r: risk-free interest rate (usually 0.02)
        q: dividend yield (usually 0)
        model: 'black_scholes' or 'black_scholes_merton'
        on_error: 'raise', 'warn', 'ignore'
        return_as: 'numpy' or 'series' or 'dataframe'
        """
        return vectorized_implied_volatility(flag, S, K, t, r, q, model=model, on_error=on_error, return_as=return_as)
    
    def delta(self, flag, S, K, t, r, sigma, q=None, model="black_scholes", return_as="numpy"):
        """
        flag: 'c' or 'p' (call or put)
        S: stock price
        K: strike price
        t: time to expiration (in years)
        r: risk-free interest rate (usually 0.02)
        q: dividend yield (usually 0)
        model: 'black', 'black_scholes', 'black_scholes_merton'
        sigma: volatility of the underlying asset
        return_as: 'numpy' or 'series' or 'dataframe'
        """
        return vectorized_delta(flag, S, K, t, r, sigma, q=q, model=model, return_as=return_as)
    
    def vega(self, flag, S, K, t, r, sigma, q=None, model="black_scholes", return_as="numpy"):
        """
        flag: 'c' or 'p' (call or put)
        S: stock price
        K: strike price
        t: time to expiration (in years)
        r: risk-free interest rate (usually 0.02)
        sigma: volatility of the underlying asset
        q: dividend yield (usually 0)
        model: 'black', 'black_scholes', 'black_scholes_merton'
        return_as: 'numpy' or 'series' or 'dataframe'
        """
        return vectorized_vega(flag, S, K, t, r, sigma, q=q, model=model, return_as=return_as)
    
    def theta(self, flag, S, K, t, r, sigma, q=None, model="black_scholes", return_as="numpy"):
        """
        flag: 'c' or 'p' (call or put)
        S: stock price
        K: strike price
        t: time to expiration (in years)
        r: risk-free interest rate (usually 0.02)
        sigma: volatility of the underlying asset
        q: dividend yield (usually 0)
        model: 'black', 'black_scholes', 'black_scholes_merton'
        return_as: 'numpy' or 'series' or 'dataframe'
        """
        return vectorized_theta(flag, S, K, t, r, sigma, q=q, model=model, return_as=return_as)
    
    def rho(self, flag, S, K, t, r, sigma, q=None, model='black_scholes', return_as="numpy"):
        """
        flag: 'c' or 'p' (call or put)
        S: stock price
        K: strike price
        t: time to expiration (in years)
        r: risk-free interest rate (usually 0.02)
        sigma: volatility of the underlying asset
        q: dividend yield (usually 0)
        model: 'black', 'black_scholes', 'black_scholes_merton'
        return_as: 'numpy' or 'series' or 'dataframe'
        """
        return vectorized_rho(flag, S, K, t, r, sigma, q=q, model=model, return_as=return_as)
    
    def gamma(self, flag, S, K, t, r, sigma, q=None, model='black_scholes', return_as="numpy"):
        """
        flag: 'c' or 'p' (call or put)
        S: stock price
        K: strike price
        t: time to expiration (in years)
        r: risk-free interest rate (usually 0.02)
        sigma: volatility of the underlying asset
        q: dividend yield (usually 0)
        model: 'black', 'black_scholes', 'black_scholes_merton'
        return_as: 'numpy' or 'series' or 'dataframe'
        """
        return vectorized_gamma(flag, S, K, t, r, sigma, q=q, model=model, return_as=return_as)
    
    def all_greeks(self, flag, S, K, t, r, sigma, q=None, model='black_scholes', return_as="numpy"):
        """
        flag: 'c' or 'p' (call or put)
        S: stock price
        K: strike price
        t: time to expiration (in years)
        r: risk-free interest rate (usually 0.02)
        sigma: volatility of the underlying asset
        q: dividend yield (usually 0)
        model: 'black', 'black_scholes', 'black_scholes_merton'
        return_as: 'dataframe' or 'json'
        """
        return get_all_greeks(flag, S, K, t, r, sigma, q=q,  model=model, return_as=return_as)
    
    def price_dataframe(df: pd.DataFrame, flag_col: str, underlying_price_col: str, strike_col: str, annualized_tte_col: str, riskfree_rate_col: str, sigma_col: str = None, price_col: str = None, dividend_col: str = None, model: str = 'black', inplace: bool = False, dtype = None) -> pd.DataFrame:
        """
        Utility function to price a DataFrame of option contracts by specifying the columns corresponding to each value.
        This function automatically calculates option price, option implied volatility and greeks in one call.
        Specifying a sigma_col will return the option prices and greeks.
        Specifying a price_col will return implied volatilities and greeks.
        Specifying both will return only greeks.

        Parameters:
        df (pd.DataFrame): Input DataFrame.
        flag_col (str): Column containing the flags ('c' for call, 'p' for puts)
        underlying_price_col (str): Column containing the price of the underlying.
        strike_col (str): Column containing the strike price.
        annualized_tte_col (str): Column containing the annualized time to expiration.
        riskfree_rate_col (str): Column containing the risk-free rate.
        sigma_col (str): Column containing the implied volatility (if unspecified, will be calculated).
        price_col (str): Column containing the price of the option (if unspecified, will be calculated).
        dividend_col (str): Column containing the implied volatility (only for Black-Scholes-Merton).
        model (str): Must be one of 'black', 'black_scholes' or 'black_scholes_merton'.
        inplace (bool): Whether to modify the input dataframe inplace (columns will be added) or return a pd.DataFrame with the result.
        dtype (dtype): Data type

        Returns:
        None if inplace is True or a pd.DataFrame object containing the desired calculations if inplace is False.
        """
        return price_dataframe(df, flag_col, underlying_price_col, strike_col, annualized_tte_col, riskfree_rate_col, sigma_col=sigma_col, price_col=price_col, dividend_col=dividend_col, model=model, inplace=inplace, dtype=dtype)