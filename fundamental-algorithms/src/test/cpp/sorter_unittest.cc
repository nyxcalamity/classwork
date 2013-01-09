/*
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
*/

#include "sorter.h"
#include "gtest/gtest.h"

//Tests BubbleSort()
TEST(BubbleSortTest, Random){
    int expected[] = { 1,2,3,4,5,6,7,8,9,10 };
    int actual[] =   { 9,7,5,3,1,8,6,4,2,10 };
    BubbleSort(actual, 10);
    for (int i=0; i<10; i++) ASSERT_EQ(expected[i],actual[i]);
}

//Tests InsertionSort()
TEST(InsertionSort, Random){
    int expected[] = { 1,2,3,4,5,6,7,8,9,10 };
    int actual[] =   { 9,7,5,3,1,8,6,4,2,10 };
    InsertionSort(actual, 10);
    for (int i=0; i<10; i++) ASSERT_EQ(expected[i],actual[i]);
}

//Tests ShellSort()
TEST(ShellSort, Random){
    int expected[] = { 1,2,3,4,5,6,7,8,9,10 };
    int actual[] =   { 9,7,5,3,1,8,6,4,2,10 };
    ShellSort(actual, 10);
    for (int i=0; i<10; i++) ASSERT_EQ(expected[i],actual[i]);
}

//Tests SelectionSort()
TEST(SelectionSort, Random){
    int expected[] = { 1,2,3,4,5,6,7,8,9,10 };
    int actual[] =   { 9,7,5,3,1,8,6,4,2,10 };
    SelectionSort(actual, 10);
    for (int i=0; i<10; i++) ASSERT_EQ(expected[i],actual[i]);
}

//Tests HeapSort()
TEST(HeapSort, Random){
    int expected[] = { 1,2,3,4,5,6,7,8,9,10 };
    int actual[] =   { 9,7,5,3,1,8,6,4,2,10 };
    HeapSort(actual, 10);
    for (int i=0; i<10; i++) ASSERT_EQ(expected[i],actual[i]);
}

//Tests MergeSort()
TEST(MergeSort, Random){
    int expected[] = { 1,2,3,4,5,6,7,8,9,10 };
    int actual[] =   { 9,7,5,3,1,8,6,4,2,10 };
    MergeSort(actual, 10);
    for (int i=0; i<10; i++) ASSERT_EQ(expected[i],actual[i]);
}

//Tests QuickSort()
TEST(QuickSort, Random){
    int expected[] = { 1,2,3,4,5,6,7,8,9,10 };
    int actual[] =   { 9,7,5,3,1,8,6,4,2,10 };
    QuickSort(actual, 10);
    for (int i=0; i<10; i++) ASSERT_EQ(expected[i],actual[i]);
}