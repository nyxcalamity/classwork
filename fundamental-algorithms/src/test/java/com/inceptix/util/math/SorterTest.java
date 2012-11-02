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
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;

/**
 * @author		Denys Sobchyshak (denys.sobchyshak@gmail.com)
 * @version 	1.1 (02.11.2012)
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
	/**
     * Sets up the test fixture for all tests.
     */
	@BeforeClass
	public static void setUpClass(){
		randomGenerator = new Random();

		for (int i = 0; i < LIST_SIZE; i++){
			SORTED_LIST.add(i);
			SORTED_DUPLICATE_LIST.add(i);
			SORTED_DUPLICATE_LIST.add(i);
		}
	}
	
	/**
     * Tears down the test fixture of all tests.
     */	
	@AfterClass
	public static void tearDownClass(){
		randomGenerator = null;
	}
	
	/**
     * Sets up the test fixture.
     */
	@Before
	public void setUp(){
		list = new ArrayList();

        // creates a list of ordered ints of size n, then pastes another list of ordered ints into random positions
		for (int i = 0; i < LIST_SIZE/2; i++){
			list.add(i);
		}
		for (int i = LIST_SIZE - 1; i >= LIST_SIZE/2; i--){
			list.add(randomGenerator.nextInt(LIST_SIZE/2), i);
		}
	}
	
	/**
     * Tears down the test fixture.
     */	
	@After
	public void tearDown(){
		list = null;
	}
	
	/**
	 * Tests duplicate values removal.
	 */
	@Test
	public void testDuplicatesRemoval(){
		Sorter.disposeOfDuplicates(SORTED_DUPLICATE_LIST);
		assertEquals("Duplicated values were removed",
                SORTED_LIST.toString(), SORTED_DUPLICATE_LIST.toString());
	}
	
	/**
	 * Tests bubble sort algorithm.
	 */
	@Test
	public void testBubbleSort(){
		assertEquals("Bubble sort procedeed incorrectly",
                SORTED_LIST.toString(), Sorter.bubbleSort(list, true).toString());
	}
	
	/**
	 * Tests quick sort algorithm.
	 */
	@Test
	public void testQuickSort(){
		assertEquals("Quick sort procedeed incorrectly",
                SORTED_LIST.toString(), Sorter.quickSort(list, true).toString());
	}
	
	/**
	 * Tests heap sort algorithm.
	 */
	@Test
	public void testHeapSort(){
        assertEquals("Heap sort procedeed incorrectly",
                SORTED_LIST.toString(), Sorter.heapSort(list, true).toString());
	}
}