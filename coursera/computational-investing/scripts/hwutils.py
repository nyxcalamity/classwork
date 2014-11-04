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
import sys
import os

import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.DataAccess as da


def get_nyse_days(start_date, end_date):
    return du.getNYSEdays(start_date, end_date, dt.timedelta(hours=16))


def load_market_data(dates, symbols, keys_to_load=tuple(['open', 'high', 'low', 'close', 'volume', 'actual_close'])):
    yahoo_data = da.DataAccess('Yahoo').get_data(dates, symbols, keys_to_load)
    return dict(zip(keys_to_load, yahoo_data))


def load_index_symbols(index="sp5002012"):
    return da.DataAccess('Yahoo').get_symbols_from_list(index)


def find_data_file(file_name):
    return os.path.join(sys.path[0], os.pardir, 'data', file_name)


def print_comparison_report(start_dt, end_dt, mean_daily_return, total_return, standard_deviation, sharpe_ratio):
    print("Detailed performance report:")
    print("Date range: " + str(start_dt) + " to " + str(end_dt))
    print("Sharpe Ratio of the Fund: " + str(sharpe_ratio[0]))
    print("Sharpe Ratio of the Benchmark: " + str(sharpe_ratio[1]))
    print("Total Return of the Fund: " + str(total_return[0]))
    print("Total Return of the Benchmark: " + str(total_return[1]))
    print("Volatility of the Fund: " + str(standard_deviation[0]))
    print("Volatility of the Benchmark: " + str(standard_deviation[1]))
    print("Average daily return of the Fund: " + str(mean_daily_return[0]))
    print("Average daily return of the Benchmark: " + str(mean_daily_return[1]))


def print_fatal_error(message):
    print(message)
    exit()