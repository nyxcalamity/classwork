package edu.tum.cs.i1.pse;

public interface Warehouse {
	public boolean hasInventory(String item, int amount);
	public int getInventory(String item);
	public void add(String item, int amount);
	public void remove(String item, int amount);
}
