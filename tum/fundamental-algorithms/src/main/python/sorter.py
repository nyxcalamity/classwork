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
import random

__author__ = "Denys Sobchyshak"
__email__ = "denys.sobchyshak@gmail.com"

class Sorter:
    """
        Provides implementation of basic sorting algorithms.
    """
    #TODO:implement most popular sorting algos
    def __init__(self):
        pass

    @staticmethod
    def bubble_sort(list2sort):
        for i in range(0, len(list2sort)-1):
            for j in range(len(list2sort)-1, 0, -1):
                if list2sort[j-1] > list2sort[j]:
                    list2sort[j-1], list2sort[j] = list2sort[j],list2sort[j-1]
        return list2sort

    @staticmethod
    def insertion_sort(list2sort):
        for i in range(1, len(list2sort)):
            item = list2sort[i]
            iHole = i
            while iHole > 0 and list2sort[iHole-1] > item:
                list2sort[iHole] = list2sort[iHole-1]
                iHole-=1
            list2sort[iHole] = item
        return list2sort

    @staticmethod
    def quick_sort(list2sort):
        """
        Pseudo inplace quick sort randomized 2-way partition implementation.
        """
        if len(list2sort) <= 1:
            return list2sort

        pivot_idx = random.randint(0, len(list2sort)-1)
        list2sort[0], list2sort[pivot_idx] = list2sort[pivot_idx], list2sort[0]

        i = 1; k = 1
        while i < len(list2sort) and k < len(list2sort):
            if list2sort[0] > list2sort[i]:
                list2sort[k], list2sort[i] = list2sort[i], list2sort[k]
                k+=1
            i+=1
        list2sort[0], list2sort[k-1] = list2sort[k-1], list2sort[0]

        return Sorter.quick_sort(list2sort[:k]) + Sorter.quick_sort(list2sort[k:])