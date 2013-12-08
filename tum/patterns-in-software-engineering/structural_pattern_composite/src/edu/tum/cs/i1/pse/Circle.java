package edu.tum.cs.i1.pse;

import java.awt.Graphics;
import java.awt.Point;
import edu.tum.cs.i1.pse.exc.NegativeValueException;

public class Circle extends AbstractComponent {
	private Point top;
	private int width;
	private int height;

	public Circle(Point top, int width, int hegiht) {
		if (top.x < 0 || top.y < 0 || width < 0 || height < 0) {
			throw new NegativeValueException();
		} else {
			this.top = top;
			this.width = width;
			this.height = hegiht;
		}
	}
	
	/* (non-Javadoc)
	 * @see edu.tum.cs.i1.pse.AbstractComponent#draw(java.awt.Graphics)
	 */
	@Override
	public void draw(Graphics graphics) {
		graphics.drawOval(top.x, top.y, width, height);
	}
}