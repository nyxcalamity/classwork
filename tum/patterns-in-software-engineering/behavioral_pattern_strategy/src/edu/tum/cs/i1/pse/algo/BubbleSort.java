package edu.tum.cs.i1.pse.algo;

public class BubbleSort extends SortingStrategy {
	@Override
	public void performSort(int a[]) {
		int n = a.length;
		int temp;

		for (int i = 0; i < n - 1; i = i + 1)
			for (int j = n - 1; j > i; j = j - 1)
				if (a[j - 1] > (a[j])) {
					temp = a[j - 1];
					a[j - 1] = a[j];
					a[j] = temp;
				}
	}
}
