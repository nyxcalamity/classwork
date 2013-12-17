package edu.tum.cs.i1.pse;

import java.util.HashMap;
import java.util.Map;

public class WarehouseImpl implements Warehouse {

	private Map<String, Integer> inventory = new HashMap<String, Integer>();

	public void add(String item, int amount) {
		inventory.put(item, amount);
	}

	public int getInventory(String item) {
		return inventory.get(item);
	}

	public boolean hasInventory(String item, int amount) {
		return inventory.get(item) >= amount;
	}

	public void remove(String item, int amount) {
		inventory.put(item, inventory.get(item) - amount);
	}

}
