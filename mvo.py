import numpy as np
from numpy.random import random, uniform
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

#spx = pd.read_csv("./SPX_500_Historical_Constituents(08-17-2024).csv")
stocks = ['XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLK', 'XLU']

dataset = yf.download(stocks, start=datetime(2013, 10, 30), end=datetime(2024, 10, 31))['Adj Close']

def log_return(data):
    log_return = np.log(data/data.shift(1))
    return log_return[1:]

log_daily_return = log_return(dataset)
print(log_daily_return)

dataset.plot(figsize=(10,6))
plt.show()
log_daily_return.plot(figsize=(10,6))
plt.show()

def portfolio_gen(returns):
    portfolio_returns = []
    portfolio_risk = []
    portfolio_weights = []

    for i in range(NUM_PORTFOLIOS):
        w = random(len(stocks))
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_returns.append(np.sum(returns.mean()*w)*rebalance_freq)
        portfolio_risk.append(np.sqrt(np.dot(w.T, np.dot(returns.cov()*rebalance_freq, w))))

    return np.array(portfolio_returns), np.array(portfolio_risk), np.array(portfolio_weights)

def optimal_portfolio(sr, r, v, w):
    optimal_index = np.where(sr == sr.max())
    r = r[optimal_index]
    v = v[optimal_index]
    sr = sr.max()
    w = w[optimal_index]

    return r, v, sr, w

lookback = 252
rebalance_freq = 126
NUM_PORTFOLIOS = 1000
actual_returns = 0
expected_sr = 0

for i in range(lookback, len(log_daily_return), rebalance_freq):
    returns, risk, weights = portfolio_gen(log_daily_return[i:i-lookback:-1])
    r, v, sr, w = optimal_portfolio(returns/risk, returns, risk, weights)
    actual_returns += np.sum(log_daily_return[i:i+rebalance_freq].mean()*w[0])*rebalance_freq
    expected_sr += sr

print("Annualized Log Returns: ", actual_returns/4)
print("Annualized Returns: ", np.exp(actual_returns/4)-1)
print("Average Expected Sharpe Ratio: ", expected_sr/8)

