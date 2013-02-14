/*
 * Copyright 2012 Denys Sobchyshak
 * 
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */
package com.inceptix.util.math;

import org.junit.*;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;

/**
 * @author  Denys Sobchyshak (denys.sobchyshak@gmail.com)
 */
public class SorterTest {
    // --- Fields ---
    private static final int LIST_SIZE = 20;
    private static final List<Integer> SORTED_LIST = new ArrayList<Integer>();
    private static final List<Integer> SORTED_DUPLICATE_LIST = new ArrayList<Integer>();

    private List<Integer> list = null;
    private static Random randomGenerator = null;

    // --- Constructors ---

    // --- Methods ---
    @BeforeClass
    public static void setUpClass(){
        randomGenerator = new Random();

        for (int i = 0; i < LIST_SIZE; i++){
            SORTED_LIST.add(i);
            SORTED_DUPLICATE_LIST.add(i);
            SORTED_DUPLICATE_LIST.add(i);
        }
    }

    @AfterClass
    public static void tearDownClass(){
        randomGenerator = null;
    }

    @Before
    public void setUp(){
        list = new ArrayList();
        for (int i = 0; i < LIST_SIZE; i++){
            list.add(i);
        }
        Collections.shuffle(list);
    }

    @After
    public void tearDown(){
        list = null;
    }

    @Test
    public void testDuplicatesRemoval(){
        Sorter.disposeOfDuplicates(SORTED_DUPLICATE_LIST);
        assertEquals("Duplicates removal failed", SORTED_LIST.toString(), SORTED_DUPLICATE_LIST.toString());
    }

    @Test
    public void testBubbleSort(){
        assertEquals("Bubble sort failed", SORTED_LIST.toString(), Sorter.bubbleSort(list, true).toString());
    }

    @Test
    public void testInsertionSort(){
        assertEquals("Insertion sort failed", SORTED_LIST.toString(), Sorter.insertionSort(list, true).toString());
    }

    @Test
    public void testQuickSort(){
        assertEquals("Quick sort failed", SORTED_LIST.toString(), Sorter.quickSort(list, true).toString());
    }

    @Test
    public void testHeapSort(){
        assertEquals("Heap sort failed", SORTED_LIST.toString(), Sorter.heapSort(list, true).toString());
    }
}