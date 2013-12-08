package edu.tum.cs.i1.pse.algo;

public class QuickSort extends SortingStrategy {

	private int[] a;
	private int n;

	@Override
	public void performSort(int[] a) {
		this.a = a;
		n = a.length;
		if (n != 0)
			quicksort(0, n - 1);
	}

	private void quicksort(int lo, int hi) {
		int i = lo, j = hi;

		// compare element x
		int x = a[(lo + hi) / 2];

		// divide
		while (i <= j) {
			while (a[i] < x)
				i++;
			while (a[j] > x)
				j--;
			if (i <= j) {
				exchange(i, j);
				i++;
				j--;
			}
		}

		// recursion
		if (lo < j)
			quicksort(lo, j);
		if (i < hi)
			quicksort(i, hi);
	}

	private void exchange(int i, int j) {
		int t = a[i];
		a[i] = a[j];
		a[j] = t;
	}

}
