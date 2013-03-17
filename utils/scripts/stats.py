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

def main():
    data = [1,17,10,12]
    print 'Arithmetic mean: ' + str(arithmeticMean(data))
    print 'Variance (sample): ' + str(variance(data))
    print 'Variance (population): ' + str(variance(data, True))

def arithmeticMean(data):
    return sum(data)/float(len(data))

def variance(data, isPopulation=False):
    avg = arithmeticMean(data)
    s = 0.0
    for i in data:
        s += (i-avg)**2
    return s / (len(data) if isPopulation else len(data)-1)

if  __name__ == '__main__':main()