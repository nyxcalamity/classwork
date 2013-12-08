/**
 * Copyright 2013 Denys Sobchyshak
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package edu.tum.cs.i1.pse;

import edu.tum.cs.i1.pse.algo.BubbleSort;
import edu.tum.cs.i1.pse.algo.MergeSort;
import edu.tum.cs.i1.pse.algo.QuickSort;
import edu.tum.cs.i1.pse.algo.SortingStrategy;

/**
 * TODO:add type description
 *
 * @author Denys Sobchyshak (denys.sobchyshak@gmail.com)
 */
public class Context {
	private SortingStrategy sorter;
	
	public Context() {
		sorter = new QuickSort();
	}
	
	public void setSortingAlgo(String algo){
		if ("quick".equals(algo))
			if (!(sorter instanceof QuickSort))
				sorter = new QuickSort();
		else if ("merge".equals(algo))
			sorter = new MergeSort();
		else 
			sorter = new BubbleSort();
	}

	public void sort(int[] array){
		sorter.performSort(array);
	}
}
