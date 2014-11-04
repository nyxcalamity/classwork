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

import numpy as np
import copy as cp
import csv

from hwutils import *


def main():
    dt_start = dt.datetime(2008, 01, 01)
    dt_end = dt.datetime(2009, 12, 31)
    analyze_period(dt_start, dt_end, 5.0, "sp5002012", "hw4-orders.csv")


def analyze_period(dt_start, dt_end, price_threshold, index, orders_file):
    symbols = load_index_symbols(index)
    symbols.append('SPY')
    dates = get_nyse_days(dt_start, dt_end)
    data = load_market_data(dates, symbols)
    data = clean_data(data, ['open', 'high', 'low', 'close', 'volume', 'actual_close'])

    num_events, event_map = find_event(symbols, data, price_threshold)

    #write data to file
    with open(find_data_file(orders_file), 'w') as f:
        writer = csv.writer(f)
        for row_idx in event_map.index:
            for column_idx in event_map.columns:
                if event_map.loc[row_idx, column_idx] == 1:
                    sell_date = row_idx+dt.timedelta(days=5)
                    sell_date = dt_end if sell_date > dt_end else sell_date
                    writer.writerow([row_idx.year, row_idx.month, row_idx.day, column_idx, 'BUY', 100])
                    writer.writerow([sell_date.year, sell_date.month, sell_date.day, column_idx, 'SELL', 100])


def clean_data(data, keys):
    for key in keys:
        data[key] = data[key].fillna(method='ffill')
        data[key] = data[key].fillna(method='bfill')
        data[key] = data[key].fillna(1.0)
    return data


def find_event(symbols, data, price_threshold):
    df_actual_close = data['actual_close']
    df_events = cp.deepcopy(df_actual_close) * np.NaN
    dt_timestamps = df_actual_close.index
    num_events = 0

    for symbol in symbols:
        for i in range(1, len(dt_timestamps)):
            price_today = df_actual_close[symbol].ix[dt_timestamps[i]]
            price_yesterday = df_actual_close[symbol].ix[dt_timestamps[i - 1]]

            if price_yesterday >= price_threshold > price_today:
                df_events[symbol].ix[dt_timestamps[i]] = 1
                num_events += 1

    return num_events, df_events


if __name__ == '__main__': main()