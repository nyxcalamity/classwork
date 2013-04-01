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
import QSTK.qstkutil.DataAccess as da
import QSTK.qstkstudy.EventProfiler as ep

import datetime as dt
import numpy as np
import copy as cp

def main():
    dt_start = dt.datetime(2008,01,01)
    dt_end = dt.datetime(2009,12,31)
    dt_timestamps = du.getNYSEdays(dt_start, dt_end, dt.timedelta(hours=16))
    price_threshold = 10.0
    keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']

    data_source = da.DataAccess('Yahoo')

    #Analysis of S&P500 2008
    symbols = data_source.get_symbols_from_list("sp5002008")
    symbols.append('SPY')
    data = cleanUpData(dict(zip(keys, data_source.get_data(dt_timestamps, symbols, keys))), keys)
    num_events, event_map = findEvent(symbols, data, price_threshold)
    print 'S&P500 in 2008 (price threshold=$'+str(price_threshold)+'): '+str(num_events)+' events ' \
        '('+dt_start.strftime("%B %d, %Y")+' to '+dt_end.strftime("%B %d, %Y")+').'
    #Note: quantities in printout don't match with those of pdfs (quizes require pdf numbers)
    ep.eventprofiler(event_map, data, i_lookback=20, i_lookforward=20,s_filename='EventStudy2008.pdf',
        b_market_neutral=True, b_errorbars=True,s_market_sym='SPY')

    #Analysis of S&P500 2012
    symbols = data_source.get_symbols_from_list("sp5002012")
    symbols.append('SPY')
    data = cleanUpData(dict(zip(keys, data_source.get_data(dt_timestamps, symbols, keys))), keys)
    num_events, event_map = findEvent(symbols, data, price_threshold)
    print 'S&P500 in 2012 (price threshold=$'+str(price_threshold)+'): '+str(num_events)+' events '\
         '('+dt_start.strftime("%B %d, %Y")+' to '+dt_end.strftime("%B %d, %Y")+').'
    #Note: quantities in printout don't match with those of pdfs (quizes require pdf numbers)
    ep.eventprofiler(event_map, data, i_lookback=20, i_lookforward=20,s_filename='EventStudy2012.pdf',
        b_market_neutral=True, b_errorbars=True,s_market_sym='SPY')

def cleanUpData(data, keys):
    for key in keys:
        data[key] = data[key].fillna(method = 'ffill')
        data[key] = data[key].fillna(method = 'bfill')
        data[key] = data[key].fillna(1.0)
    return data

def findEvent(symbols, data, price_threshold):
    df_actual_close = data['actual_close']
    df_events = cp.deepcopy(df_actual_close)*np.NAN
    dt_timestamps = df_actual_close.index
    num_events = 0

    for symbol in symbols:
        for i in range(1, len(dt_timestamps)):
            price_today = df_actual_close[symbol].ix[dt_timestamps[i]]
            price_yesterday = df_actual_close[symbol].ix[dt_timestamps[i - 1]]

            if price_yesterday >= price_threshold and price_today < price_threshold:
                df_events[symbol].ix[dt_timestamps[i]] = 1
                num_events+=1

    return num_events, df_events


if __name__ == '__main__':main()