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
__author__ = 'Denys Sobchyshak'
__email__ = 'denys.sobchyshak@gmail.com'


def geom_series(z, t):
    return (1.0-z**(t+1))/(1.0-z)


def d(t1=0, t2=1, r=0.1, n=1, spot=None):
    if not spot:
        return (1+r/n)**(-(t2-t1)*n)
    else:
        return (1+spot[t2-1])**(-t2)


def f_rate(t1=1, t2=2, spot=[0.063, 0.069]):
    return ((1+spot[t2-1])**t2/(1+spot[t1-1])**t1)**(1/(t2-t1))-1