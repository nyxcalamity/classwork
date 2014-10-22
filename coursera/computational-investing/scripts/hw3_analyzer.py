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
import csv

import QSTK.qstkutil.tsutil as tsu

from hwutils import *


def analyze_fund(values_file, benchmark):
    # read orders file
    fund_dates = []
    fund_close = []
    with open(values_file, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:
            fund_dates.append(dt.datetime(int(row[0]), int(row[1]), int(row[2]), 16))
            fund_close.append(float(row[3]))

    start_dt = fund_dates[0]
    end_dt = fund_dates[-1]

    #compute fund statistics
    fund_daily_returns = np.array(fund_close)/fund_close[0]
    tsu.returnize0(fund_daily_returns)
    fund_mean_return = sum(fund_daily_returns)/len(fund_daily_returns)
    fund_cumulative_return = fund_close[-1]/fund_close[0]-1
    fund_returns_volatility = np.sqrt(sum((fund_daily_returns-fund_mean_return)**2)/len(fund_daily_returns))
    fund_sharpe_ratio = np.sqrt(252)*fund_mean_return/fund_returns_volatility

    #compute benchmark statistics
    analysis_period = get_nyse_days(start_dt, end_dt)
    benchmark_close = load_market_data(analysis_period, [benchmark])['actual_close']
    benchmark_close = benchmark_close[benchmark].values
    benchmark_daily_returns = benchmark_close/benchmark_close[0]
    tsu.returnize0(benchmark_daily_returns)
    benchmark_mean_return = sum(benchmark_daily_returns)/len(benchmark_daily_returns)
    benchmark_cumulative_return = benchmark_close[-1]/benchmark_close[0]-1
    benchmark_return_volatility = np.sqrt(sum((benchmark_daily_returns-benchmark_mean_return)**2)/len(benchmark_daily_returns))
    benchmark_sharpe_ratio = np.sqrt(252)*benchmark_mean_return/benchmark_return_volatility

    print_comparison_report(start_dt, end_dt,
                            (fund_mean_return, benchmark_mean_return),
                            (fund_cumulative_return, benchmark_cumulative_return),
                            (fund_returns_volatility, benchmark_return_volatility),
                            (fund_sharpe_ratio, benchmark_sharpe_ratio))


def process_cli():
    if len(sys.argv) < 2:
        print_fatal_error("Please provide more arguments in the format: values_file benchmark")

    values_file = find_data_file(sys.argv[1])
    benchmark = sys.argv[2]

    if not os.path.isfile(values_file) or not os.access(values_file, os.R_OK):
        print_fatal_error("Provided values file doesn't exist or it can't be read: " + values_file)

    return values_file, benchmark


if __name__ == '__main__':
    a, b = process_cli()
    analyze_fund(a, b)
