#!/usr/bin/env python
"""
    Copyright 2013 Denys Sobchyshak

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

__author__ = "Denys Sobchyshak"
__email__ = "denys.sobchyshak@gmail.com"

import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

# Third Party Imports
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def main():
    sd,avg_dreturns = assesPerformance(dt.datetime(2011,1,1), dt.datetime(2011,12,31), ['AAPL', 'GLD', 'GOOG', 'XOM'], [0.4, 0.4, 0.0, 0.2])
    print 'Standard deviation: ' + str(sd)
    print 'Average daily returns: ' + str(avg_dreturns)

def assesPerformance(start_date, end_date, symbols, allocation):
    if len(symbols) != len(allocation):
        return None, None
    if sum(allocation) != 1:
        return None, None

    n_symbols = len(symbols)

    #Arranging timestamps
    time_of_day = dt.timedelta(hours=16)
    days_from_nyse = du.getNYSEdays(start_date, end_date, time_of_day)

    #Loading data from Yahoo
    keys_to_load = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    yahoo_data_source = da.DataAccess('Yahoo')
    yahoo_data = yahoo_data_source.get_data(days_from_nyse, symbols, keys_to_load)
    data = dict(zip(keys_to_load, yahoo_data))

    close_price = data['close'].values
    n_entries = close_price.size

    #Computing standard deviation
    avg = np.arange(n_symbols)
    avg[:] = sum(close_price[:,:])/close_price.shape[0]
    std_dev = np.ones(n_symbols)
    std_dev[:] = np.sqrt(sum((close_price[:,:]-avg[:])**2)/n_entries)

    #Computing daily returns
    daily_returns = close_price.copy()
    daily_returns[1:,:] = (daily_returns[1:,:]/daily_returns[0:-1]) - 1
    daily_returns[0] = np.zeros(daily_returns.shape[1])

    #Computing average daily returns
    avg_daily_returns = np.zeros(n_symbols)
    avg_daily_returns[:] = sum(daily_returns[:])/daily_returns.shape[0]

    return std_dev, avg_daily_returns


if __name__ == '__main__':
    main()