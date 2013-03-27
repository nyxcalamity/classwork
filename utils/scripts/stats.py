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
    data = [10,7,4,6,9,4]
    print 'Mode: ' + str(mode(data))
    print 'Median: ' + str(median(data))
    print 'Arithmetic mean: ' + str(arithmeticMean(data))
    print 'Variance (sample): ' + str(variance(data))
    print 'Variance (population): ' + str(variance(data, True))

def mode(data):
    """
    Uses counting sort to figure out the mode.
    """
    count = [0]*(max(data)+1)
    for i in data:
        count[i] += 1
    return count.index(max(count))

def median(data):
    data.sort()
    m = len(data)/2
    return data[m] if len(data)%2 > 0 else float((data[m]+data[m-1]))/2

def arithmeticMean(data):
    return sum(data)/float(len(data))

def variance(data, isPopulation=False):
    avg = arithmeticMean(data)
    se = 0.0
    for i in data:
        se += (i-avg)**2
    return se / (len(data) if isPopulation else len(data)-1)

if  __name__ == '__main__':main()