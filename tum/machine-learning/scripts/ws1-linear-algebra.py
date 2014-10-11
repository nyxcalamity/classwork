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

from numpy import *

def main():
    A = array([2, -1, 0, -1, 2, -1, 0, -1, 2]).reshape(3, 3)
    w,v = linalg.eig(A)
    print ('Eigenvalues: \n'+str(w))
    print ('Eigenvectors: \n'+str(v))

if __name__ == '__main__':main()