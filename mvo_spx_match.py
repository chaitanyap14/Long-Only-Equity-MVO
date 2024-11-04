import numpy as np
from numpy.random import random, uniform
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

spx = pd.read_csv("./SPX_top20.csv")
spx['tickers'] = spx['tickers'].apply(lambda x: x.split(', '))
spx['date'] = pd.to_datetime(spx['date'])

print(spx['date'][34], spx['date'][34]+timedelta(days=182))

lookback = 378
rebalance_freq = 126

for i in range(len(spx)-1):
    print(spx['date'][i], spx['tickers'][i])
    print(spx['date'][i]+timedelta(days=182), spx['tickers'][i])

