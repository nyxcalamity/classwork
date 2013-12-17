package edu.tum.cs.i1.pse;

public class OrderImpl implements Order {
	private String item;
	private int amount;
	private boolean filled = false;

	public OrderImpl(String item, int amount) {
		this.item = item;
		this.amount = amount;
	}

	public void fillOut(Warehouse warehouse) {
		if(warehouse.hasInventory(item, amount)) {
			warehouse.remove(item, amount);
			filled = true;
		}
	}

	public boolean isFilled() {
		return filled;
	}

}
