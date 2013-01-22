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
#include "sorter.h"
#include "gtest/gtest.h"

//default size
const int _size = 100;

//Random number generator
int RandomNumber(){ return std::rand()%(2*_size); }

//Generates a sorted vector
std::vector<int> GetSortedVector(){
    //init vector
    std::vector<int> v(_size);
    //fill with random numbers
    generate(v.begin(), v.end(), RandomNumber);
    //sort the vector
    RadixSort(&v[0],v.size(), 2*_size);
    //return the setup vector
    return v;
}

//Tests BinarySearch()
TEST(BinarySearch, NonUniqueRandomSorted){
    std::vector<int> v = GetSortedVector();
    int expected_key = std::rand()%_size;

    int actual_key = BinarySearch(v, v[expected_key]);
    ASSERT_EQ(v[expected_key], v[actual_key]) \
            << "Searched key(" << expected_key << ") didn't match returned key(" << actual_key <<")";

    actual_key = BinarySearch(v, INT_MAX);
    ASSERT_EQ(-1, actual_key)
        << "Found a non-existing value (key=" << actual_key << ")";
}