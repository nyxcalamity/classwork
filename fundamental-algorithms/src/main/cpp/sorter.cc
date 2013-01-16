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

#include <algorithm>
#include <climits>
#include "sorter.h"

//swaps two elements of an array
void swap(int array[], int i, int j){
    if (i != j) {
        int tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
    }
}

//Performs a bubble sort on an array of ints.
void BubbleSort(int *array, int array_size){
    for (int i = 0; i<array_size; i++){
        bool swapped = false;
        for (int j = array_size; j>i; j--){
            if (array[j-1] > array[j]){
                swap(array, j-1, j);
                swapped = true;
            }
        }
        //-> invariant: array[0..i] is sorted and elements are in final position
        if (!swapped) break;
    }
};

//Performs an insertion sort on an array of ints.
void InsertionSort(int *array, int array_size){
    for (int i = 1; i < array_size; i++){
        int key = array[i];
        int j = i-1;
        while (j >= 0 && array[j] > key) {
            array[j+1] = array[j];
            j--;
        }
        array[j+1] = key;
        //-> invariant: array[0..j] is sorted
    }
};

//Performs a shell sort on an array of ints.
void ShellSort(int *array, int array_size){
    int h = 1;
    while (h < array_size) h = 3*h + 1;
    while (h > 0){
        h = h/3;
        for (int k = 0; k < h; k++){
            for (int i = h+k; i < array_size; i+=h){
                int key = array[i];
                int j = i-h;
                while (j>=0 && array[j] > key){
                    array[j+h] = array[j];
                    j-=h;
                }
                array[j+h] = key;
                //-> invariant: array[k,k+h...j] is sorted
            }
        }
        //->invariant: each h-sub-array is sorted
    }
};

//Performs a selection sort on an array of ints.
void SelectionSort(int *array, int array_size){
    for (int i = 0; i < array_size; i++){
        int iMin = i;

        for(int j = i+1; j < array_size; j++){
            if (array[iMin] > array[j]) iMin = j;
        }

        if (iMin != i){
            swap(array, i, iMin);
        }
        //->invariant: array[0...i] is sorted
    }
};

//Performs a merge sort on an array of ints.
void MergeSort(int *array, int start, int end){
    if (start >= end) return;

    int mid = (start+end)/2;
    MergeSort(array,start,mid);
    MergeSort(array,mid+1,end);

    //merge operation
    int left_array[mid+1-start];
    std::copy(&array[start], &array[mid+1], left_array);

    int i=0,j=mid+1,k=start;
    while (i+start<=mid && j <= end){
        array[k++]= (left_array[i] < array[j]) ? left_array[i++] : array[j++];
        //->invariant: a[start..k] in final position
    }

    while (i+start<=mid) {
        array[k++] = left_array[i++];
        //->invariant: a[start..k] in final position
    }
};

//Performs a heap sort on an array of ints.
void HeapSort(int *array, int array_size){};

//Performs a quick sort on an array of ints.
void QuickSort(int *array, int start, int end){
    if (start >= end) return;

    //2-way partition
    int k=0;
    for (int i=1; i <= end; i++){
        if (array[i] < array[1])
            swap(array, ++k, i);
    }
    swap(array, start, k);
    //->invariant: array[start..k-1] < array[k] <= array[k+1..end]

    QuickSort(array, start, k-1);
    QuickSort(array, k+1, end);
};