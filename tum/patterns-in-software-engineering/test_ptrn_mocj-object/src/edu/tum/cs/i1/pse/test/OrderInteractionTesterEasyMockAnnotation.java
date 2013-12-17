package edu.tum.cs.i1.pse.test;

import static org.easymock.EasyMock.*;
import static org.junit.Assert.*;

import org.easymock.EasyMockRunner;
import org.easymock.TestSubject;
import org.easymock.Mock;
import org.junit.Test;
import org.junit.runner.RunWith;

import edu.tum.cs.i1.pse.Order;
import edu.tum.cs.i1.pse.OrderImpl;
import edu.tum.cs.i1.pse.Warehouse;

@RunWith(EasyMockRunner.class)
public class OrderInteractionTesterEasyMockAnnotation {
	
	private static String TALISKER = "Talisker";

	@TestSubject
    private Order order = new OrderImpl(TALISKER, 50);
	
	@Mock
	private Warehouse warehouseMock;
	
	@Test
	public void fillingRemovesInventoryIfInStock() {
		expect(warehouseMock.hasInventory(TALISKER, 50)).andReturn(true);
		warehouseMock.remove(TALISKER, 50);
		replay(warehouseMock);
		order.fillOut(warehouseMock);
		assertTrue(order.isFilled());
		verify(warehouseMock);
	}
	
	@Test
	public void fillingDoesNotRemoveIfNotEnoughInStock(){
		order = new OrderImpl(TALISKER, 51);
		warehouseMock = createMock(Warehouse.class);
		
		expect(warehouseMock.hasInventory(TALISKER, 51)).andReturn(false);
		replay(warehouseMock);
		
		order.fillOut(warehouseMock);
		assertFalse(order.isFilled());
		
		verify(warehouseMock);
	}
}
