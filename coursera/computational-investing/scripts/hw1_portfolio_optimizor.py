#!/usr/bin/env python
"""
    Copyright 2014 Denys Sobchyshak

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

import datetime as dt
import numpy as np


#-----------------------------------------------------------------------------------------------------------------------
# Computational logic
#-----------------------------------------------------------------------------------------------------------------------
def assesPerformance(start_date, end_date, symbols, allocations):
    if len(symbols) != len(allocations):
        return None
    if round(sum(allocations)) > 1: #rounding due to floating-point arithmetics approximation errors
        return None

    #loading the holding period
    holding_period = du.getNYSEdays(start_date, end_date, dt.timedelta(hours=16))

    #Loading data from Yahoo
    keys_to_load = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    yahoo_data_source = da.DataAccess('Yahoo')
    yahoo_data = yahoo_data_source.get_data(holding_period, symbols, keys_to_load)
    data = dict(zip(keys_to_load, yahoo_data))

    close_price = data['close'].values
    n_entries = close_price.shape[0]

    #calculating portfolio close values
    tmp = close_price/close_price[0]
    tmp = tmp*allocations
    portfolio_close_price = np.zeros(n_entries)
    for i in range(n_entries):
        portfolio_close_price[i] = sum(tmp[i])

    #Computing daily returns
    daily_returns = portfolio_close_price.copy()
    tsu.returnize0(daily_returns) #computes daily returns
    daily_returns_avg = sum(daily_returns)/n_entries

    #Computing cumulative return
    cumulative_return = portfolio_close_price[-1]/portfolio_close_price[0]

    #Computing volatility (standard deviation)
    volatility = np.sqrt(sum((daily_returns[:]-daily_returns_avg)**2)/n_entries)

    #Computing sharpe ratio
    sharpe_ratio = np.sqrt(252)*daily_returns_avg/volatility

    return daily_returns_avg, cumulative_return, volatility, sharpe_ratio


def optimizePortfolio(start_date, end_date, symbols):
    optimal_portfolio = ([],0,0,0,0) #tuple of (allocations, daily_rets_avg, cum_rets, volatility, sharpe_ratio)
    for i in range(11):
        print str(i)+'0% done'
        for j in range(11):
            for k in range(11):
                for m in range(11):
                    if i+j+k+m == 10:
                        allocations=np.array([i,j,k,m])/10.0
                        daily_rets_avg, cum_rets, volatility, sharpe_ratio = assesPerformance(start_date, end_date, symbols, allocations)
                        if sharpe_ratio > optimal_portfolio[-1]:
                            optimal_portfolio = (allocations, daily_rets_avg, cum_rets, volatility, sharpe_ratio)
    return optimal_portfolio


def displayResults(start_date,end_date,symbols,allocations,sharpe,volatility,avg_daily_rets,cum_rets):
    print 'Start Date: ' + start_date.strftime("%B %d, %Y")
    print 'End Date: ' + end_date.strftime("%B %d, %Y")
    print 'Symbols: ' + str(symbols)
    print 'Optimal Allocations: ' + str(allocations)
    print 'Sharpe Ratio: ' + str(sharpe)
    print 'Volatility (standard deviation of daily returns): ' + str(volatility)
    print 'Average Daily Return: ' + str(avg_daily_rets)
    print 'Cumulative Return: ' + str(cum_rets)
    print '\n\n'

#-----------------------------------------------------------------------------------------------------------------------
# Test data
#-----------------------------------------------------------------------------------------------------------------------
def test():
    #First example
    start_date = dt.datetime(2011,1,1)
    end_date = dt.datetime(2011,12,31)
    symbols = ['AAPL', 'GLD', 'GOOG', 'XOM']
    allocations = [0.4, 0.4, 0.0, 0.2]
    avg_daily_rets, cum_rets, volatility, sharpe = assesPerformance(start_date, end_date, symbols, allocations)
    displayResults(start_date,end_date,symbols,allocations,sharpe,volatility,avg_daily_rets,cum_rets)

#    Expected results:
#    Start Date: January 1, 2011
#    End Date: December 31, 2011
#    Symbols: ['AAPL', 'GLD', 'GOOG', 'XOM']
#    Optimal Allocations: [0.4, 0.4, 0.0, 0.2]
#    Sharpe Ratio: 1.02828403099
#    Volatility (stdev of daily returns):  0.0101467067654
#    Average Daily Return:  0.000657261102001
#    Cumulative Return:  1.16487261965

    #Second example
    start_date = dt.datetime(2010,1,1)
    end_date = dt.datetime(2010,12,31)
    symbols = ['AXP', 'HPQ', 'IBM', 'HNZ']
    allocations = [0.0, 0.0, 0.0, 1.0]
    avg_daily_rets, cum_rets, volatility, sharpe = assesPerformance(start_date, end_date, symbols, allocations)
    displayResults(start_date,end_date,symbols,allocations,sharpe,volatility,avg_daily_rets,cum_rets)

#    Expected results:
#    Start Date: January 1, 2010
#    End Date: December 31, 2010
#    Symbols: ['AXP', 'HPQ', 'IBM', 'HNZ']
#    Optimal Allocations:  [0.0, 0.0, 0.0, 1.0]
#    Sharpe Ratio: 1.29889334008
#    Volatility (stdev of daily returns): 0.00924299255937
#    Average Daily Return: 0.000756285585593
#    Cumulative Return: 1.1960583568

#-----------------------------------------------------------------------------------------------------------------------
# Quizzes
#-----------------------------------------------------------------------------------------------------------------------
def quiz():
    start_date = dt.datetime(2010,1,1)
    end_date = dt.datetime(2010,12,31)
    symbols = ['BRCM', 'ADBE', 'AMD', 'ADI']
    optimal_portfolio = optimizePortfolio(start_date, end_date, symbols)

    print '\nOptimal portfolio would be: '
    displayResults(start_date, end_date, symbols, optimal_portfolio[0], optimal_portfolio[-1], optimal_portfolio[-2],\
        optimal_portfolio[1], optimal_portfolio[2])


if __name__ == '__main__':
    #test()
    quiz()
