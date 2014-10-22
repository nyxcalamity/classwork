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

import datetime as dt
import numpy as np
import pandas as ps
import csv

import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

from hwutils import *


def main(cash_balance, orders_file, values_file):
    # read orders file
    orders = {}
    with open(orders_file) as f:
        reader = csv.reader(f)
        for row in reader:
            date = dt.datetime(int(row[0]), int(row[1]), int(row[2]), 16)
            if date in orders:
                orders[date].append(row[3:])
            else:
                orders[date] = [row[3:]]

    #extract data
    order_dates = sorted(orders.keys())
    symbols = set()
    for v in orders.values():
        for order in v:
            symbols.add(order[0])
    symbols = sorted(symbols)

    #load active dates
    dates = get_nyse_days(order_dates[0], order_dates[-1]+dt.timedelta(days=1))

    #load price matrix
    market_data = load_market_data(dates, symbols)
    price_df = market_data['close']

    #generate trade matrix
    trade_df = ps.DataFrame(np.zeros((len(dates), len(symbols))), index=dates, columns=symbols)
    for k, v in orders.iteritems():
        for order in v:
            trade_df.at[k, order[0]] = int(order[2]) if order[1] == "Buy" else -int(order[2])

    #initialize cache balance series
    cash_ts = ps.Series([0]*(len(dates)), index=dates)
    cash_ts[0] = cash_balance - trade_df.ix[0].dot(price_df.ix[0])
    for i in range(1, len(dates)):
        cash_ts[i] = -trade_df.ix[i].dot(price_df.ix[i])

    #creating value series
    price_df['_CASH'] = 1.0
    trade_df['_CASH'] = cash_ts
    trade_df = trade_df.cumsum()
    fund_ts = ps.Series([0]*(len(dates)), index=dates)
    for i in range(0, len(dates)):
        fund_ts[i] = trade_df.ix[i].dot(price_df.ix[i])

    #write data to file
    with open(values_file, 'w') as f:
        writer = csv.writer(f)
        for row_idx in fund_ts.index:
            writer.writerow([row_idx.year, row_idx.month, row_idx.day, fund_ts[row_idx]])


def get_nyse_days(start_date, end_date):
    return du.getNYSEdays(start_date, end_date, dt.timedelta(hours=16))


def load_market_data(dates, symbols, keys_to_load=tuple(['open', 'high', 'low', 'close', 'volume', 'actual_close'])):
    yahoo_data = da.DataAccess('Yahoo').get_data(dates, symbols, keys_to_load)
    return dict(zip(keys_to_load, yahoo_data))


def process_cli():
    if len(sys.argv) < 3:
        print_fatal_error("Please provide more arguments in the format: cache_balance orders_file values_file")

    cache_balance = sys.argv[1]
    orders_file = find_data_file(sys.argv[2])
    values_file = find_data_file(sys.argv[3])

    if not os.path.isfile(orders_file) or not os.access(orders_file, os.R_OK):
        print_fatal_error("Provided orders file doesn't exist or it can't be read: " + orders_file)
    with open(values_file, 'w') as f:
        if not os.path.isfile(f.name):
            print_fatal_error("Couldn't create values file: " + values_file)

    return int(cache_balance), orders_file, values_file


if __name__ == '__main__':
    a, b, c = process_cli()
    main(a, b, c)