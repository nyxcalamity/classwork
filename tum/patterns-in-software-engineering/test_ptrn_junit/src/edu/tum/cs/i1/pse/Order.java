package edu.tum.cs.i1.pse;

public interface Order {
	public boolean isFilled();
	public void fill(Warehouse warehouse);
}
