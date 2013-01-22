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

int _array_size = 100;
int _max = 300;

void FillNonUniqueRandomNumbers(int array[], int max_value){
    for (int i=0; i<_array_size; i++)
        array[i] = std::rand()%max_value;
}

//Tests BubbleSort()
TEST(BubbleSortTest, NonUniqueRandom){
    int a[_array_size];
    FillNonUniqueRandomNumbers(a, INT_MAX);
    BubbleSort(a, _array_size);
    for (int i=1; i<_array_size; i++) ASSERT_TRUE(a[i-1] <= a[i]);
}

//Tests InsertionSort()
TEST(InsertionSort, NonUniqueRandom){
    int a[_array_size];
    FillNonUniqueRandomNumbers(a, INT_MAX);
    InsertionSort(a, _array_size);
    for (int i=1; i<_array_size; i++) ASSERT_TRUE(a[i-1] <= a[i]);
}

//Tests ShellSort()
TEST(ShellSort, NonUniqueRandom){
    int a[_array_size];
    FillNonUniqueRandomNumbers(a, INT_MAX);
    ShellSort(a, _array_size);
    for (int i=1; i<_array_size; i++) ASSERT_TRUE(a[i-1] <= a[i]);
}

//Tests SelectionSort()
TEST(SelectionSort, NonUniqueRandom){
    int a[_array_size];
    FillNonUniqueRandomNumbers(a, INT_MAX);
    SelectionSort(a, _array_size);
    for (int i=1; i<_array_size; i++) ASSERT_TRUE(a[i-1] <= a[i]);
}

//Tests MergeSort()
TEST(MergeSort, NonUniqueRandom){
    int a[_array_size];
    FillNonUniqueRandomNumbers(a, INT_MAX);
    MergeSort(a, 0, _array_size-1);
    for (int i=1; i<_array_size; i++) ASSERT_TRUE(a[i-1] <= a[i]);
}

//Tests QuickSort()
TEST(QuickSort, NonUniqueRandom){
    int a[_array_size];
    FillNonUniqueRandomNumbers(a, INT_MAX);
    QuickSort(a, 0, _array_size-1);
    for (int i=1; i<_array_size; i++) ASSERT_TRUE(a[i-1] <= a[i]);
}

//Tests QuickSort3()
TEST(QuickSort3, NonUniqueRandom){
    int a[_array_size];
    FillNonUniqueRandomNumbers(a, INT_MAX);
    QuickSort3(a, 0, _array_size-1);
    for (int i=1; i<_array_size; i++) ASSERT_TRUE(a[i-1] <= a[i]);
}

//Tests HeapSort()
TEST(HeapSort, NonUniqueRandom){
    int a[_array_size];
    FillNonUniqueRandomNumbers(a, INT_MAX);
    HeapSort(a, _array_size);
    for (int i=1; i<_array_size; i++)
        ASSERT_TRUE(a[i-1] <= a[i]) << a[i-1] << " is bigger than " << a[i];
}

//Tests CountingSort()
TEST(CountingSort, NonUniqueRandom){
    int a[_array_size];
    FillNonUniqueRandomNumbers(a, _max);
    CountingSort(a, _array_size, _max);
    for (int i=1; i<_array_size; i++)
        ASSERT_TRUE(a[i-1] <= a[i]) << a[i-1] << " is bigger than " << a[i];
}

//Tests RadixSort()
TEST(RadixSort, NonUniqueRandom){
    int a[_array_size];
    FillNonUniqueRandomNumbers(a, _max);
    RadixSort(a, _array_size, _max);
    for (int i=1; i<_array_size; i++)
        ASSERT_TRUE(a[i-1] <= a[i]) << a[i-1] << " is bigger than " << a[i];
}