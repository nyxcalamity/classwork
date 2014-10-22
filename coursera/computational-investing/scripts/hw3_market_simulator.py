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
import sys
import os
import csv

import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

from hwutils import *


def main():
    if len(sys.argv) < 3:
        print_fatal_error("Please provide more arguments in the format: cache_balance orders.file values.file")
    cache_balance = sys.argv[1]
    orders_file = find_data_file(sys.argv[2])
    values_file = find_data_file(sys.argv[3])

    if not os.path.isfile(orders_file) or not os.access(orders_file, os.R_OK):
        print_fatal_error("Provided orders file doesn't exist or it can't be read: " + orders_file)
    # if not os.path.isfile(values_file) or not os.access(values_file, os.W_OK):
    #     printError("Provided values file doesn't exist or it can't be written to." + values_file)

    values = []
    with open(orders_file) as values_csv:
        reader = csv.reader(values_csv)
        for row in reader:
            values.append(row)
    print values


if __name__ == '__main__':
    main()
