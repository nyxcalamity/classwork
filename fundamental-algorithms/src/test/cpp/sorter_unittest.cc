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

int array_size = 100;
int max = 300;

void FillNonUniqueRandomNumbers(int array[], int max_value){
    for (int i=0; i<array_size; i++)
        array[i] = std::rand()%max_value;
}

//Tests BubbleSort()
TEST(BubbleSortTest, NonUniqueRandom){
    int a[array_size];
    FillNonUniqueRandomNumbers(a, INT_MAX);
    BubbleSort(a, array_size);
    for (int i=1; i<array_size; i++) ASSERT_TRUE(a[i-1] <= a[i]);
}

//Tests InsertionSort()
TEST(InsertionSort, NonUniqueRandom){
    int a[array_size];
    FillNonUniqueRandomNumbers(a, INT_MAX);
    InsertionSort(a, array_size);
    for (int i=1; i<array_size; i++) ASSERT_TRUE(a[i-1] <= a[i]);
}

//Tests ShellSort()
TEST(ShellSort, NonUniqueRandom){
    int a[array_size];
    FillNonUniqueRandomNumbers(a, INT_MAX);
    ShellSort(a, array_size);
    for (int i=1; i<array_size; i++) ASSERT_TRUE(a[i-1] <= a[i]);
}

//Tests SelectionSort()
TEST(SelectionSort, NonUniqueRandom){
    int a[array_size];
    FillNonUniqueRandomNumbers(a, INT_MAX);
    SelectionSort(a, array_size);
    for (int i=1; i<array_size; i++) ASSERT_TRUE(a[i-1] <= a[i]);
}

//Tests MergeSort()
TEST(MergeSort, NonUniqueRandom){
    int a[array_size];
    FillNonUniqueRandomNumbers(a, INT_MAX);
    MergeSort(a, 0, array_size-1);
    for (int i=1; i<array_size; i++) ASSERT_TRUE(a[i-1] <= a[i]);
}

//Tests QuickSort()
TEST(QuickSort, NonUniqueRandom){
    int a[array_size];
    FillNonUniqueRandomNumbers(a, INT_MAX);
    QuickSort(a, 0, array_size-1);
    for (int i=1; i<array_size; i++) ASSERT_TRUE(a[i-1] <= a[i]);
}

//Tests QuickSort3()
TEST(QuickSort3, NonUniqueRandom){
    int a[array_size];
    FillNonUniqueRandomNumbers(a, INT_MAX);
    QuickSort3(a, 0, array_size-1);
    for (int i=1; i<array_size; i++) ASSERT_TRUE(a[i-1] <= a[i]);
}

//Tests HeapSort()
TEST(HeapSort, NonUniqueRandom){
    int a[array_size];
    FillNonUniqueRandomNumbers(a, INT_MAX);
    HeapSort(a, array_size);
    for (int i=1; i<array_size; i++)
        ASSERT_TRUE(a[i-1] <= a[i]) << a[i-1] << " is bigger than " << a[i];
}

//Tests CountingSort()
TEST(CountingSort, NonUniqueRandom){
    int a[array_size];
    FillNonUniqueRandomNumbers(a, max);
    CountingSort(a, array_size, max);
    for (int i=1; i<array_size; i++)
        ASSERT_TRUE(a[i-1] <= a[i]) << a[i-1] << " is bigger than " << a[i];
}

//Tests RadixSort()
TEST(RadixSort, NonUniqueRandom){
    int a[array_size];
    FillNonUniqueRandomNumbers(a, max);
    RadixSort(a, array_size, max);
    for (int i=1; i<array_size; i++)
        ASSERT_TRUE(a[i-1] <= a[i]) << a[i-1] << " is bigger than " << a[i];
}