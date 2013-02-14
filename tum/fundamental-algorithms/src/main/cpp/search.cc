/*
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
*/

#include "search.h"

// Performs a binary search over the provided vector. Returns -1 if value not found.
int BinarySearch(std::vector<int> a, int value){
    int start = 0, end = a.size()-1;

    while (start <= end) {
        int mid = (start+end)/2;
        if (a[mid] == value) return mid;
        if (a[mid] < value) start = mid+1;
        if (a[mid] > value) end = mid-1;
    }

    return -1;
}