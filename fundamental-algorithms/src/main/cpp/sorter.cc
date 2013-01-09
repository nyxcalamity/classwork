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

#include <iostream>
#include "sorter.h"

//Performs a bubble sort on an array of ints.
void BubbleSort(int *array, int array_size){
    for (int i = 0; i<array_size; i++){
        bool swapped = false;
        for (int j = array_size; j>i; j--){
            if (array[j-1] > array[j]){
                int tmp = array[j-1];
                array[j-1] = array[j];
                array[j] = tmp;
                swapped = true;
            }
        }
        if (!swapped) break;
    }
};

//Performs a shell sort on an array of ints.
void ShellSort(int *array, int array_size){

};

//Performs an insertion sort on an array of ints.
void InsertionSort(int *array, int array_size){

};

//Performs a selection sort on an array of ints.
void SelectionSort(int *array, int array_size){

};

//Performs a heap sort on an array of ints.
void HeapSort(int *array, int array_size){

};

//Performs a merge sort on an array of ints.
void MergeSort(int *array, int array_size){

};

//Performs a quick sort on an array of ints.
void QuickSort(int *array, int array_size){

};