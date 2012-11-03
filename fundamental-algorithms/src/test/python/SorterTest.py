#!/usr/bin/env python
"""
    Copyright 2012 Denys Sobchyshak

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

import Sorter as math

import random
import unittest

class SorterTestCase(unittest.TestCase):

    def setUp(self):
        self.sorted_sequence = range(20)
        self.unsorted_sequence = range(20)
        random.shuffle(self.unsorted_sequence)

    def tearDown(self):
        self.unsorted_sequence = []
        self.sorted_sequence = []

    def test_bubble_sort(self):
        self.assertEquals(self.sorted_sequence, math.Sorter.bubble_sort(self.unsorted_sequence))

    def test_insertion_sort(self):
        self.fail()

    def test_quick_sort(self):
        self.fail()

if __name__ == '__main__':
    unittest.main()
