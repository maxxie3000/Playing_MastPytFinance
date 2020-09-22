import yfinance as yf
import pandas as pd
import numpy as np

df = yf.download('AAPL',actions = 'inline', start='2017-01-01',
            end='2017-12-31', progress=False)

#%%
prices = df['Adj Close']

from scipy.stats import norm

def calculate_daily_var(
        portfolio, prob, mean, stdev, days_per_year=252 ):
    alpha = 1-prob
    u = mean/days_per_year
    sigma = stdev/np.sqrt(days_per_year)
    norminv = norm.ppf(alpha, u, sigma)
    return portfolio - portfolio*(norminv+1)

portfolio = 100000000
confidence = 0.95

daily_returns = prices.pct_change().dropna()
mu = np.mean(daily_returns)
sigma = np.std(daily_returns)

VaR = calculate_daily_var(portfolio, confidence, mu, sigma, days_per_year=252)
print(f'Value-at-Risk: {VaR:.2f}')
