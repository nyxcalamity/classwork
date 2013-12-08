package edu.tum.cs.i1.pse;

import java.awt.Graphics;
import java.awt.Point;

import edu.tum.cs.i1.pse.exc.NegativeValueException;

public class Line extends AbstractComponent {
	private Point startPoint, endPoint;

	public Line(Point startPoint, Point endPoint) {
		if (startPoint.x < 0 || startPoint.y < 0 || endPoint.x < 0 || endPoint.y < 0) {
			throw new NegativeValueException();
		} else {
			this.startPoint = startPoint;
			this.endPoint = endPoint;
		}
	}

	/* (non-Javadoc)
	 * @see edu.tum.cs.i1.pse.AbstractComponent#draw(java.awt.Graphics)
	 */
	@Override
	public void draw(Graphics graphics) {
		graphics.drawLine(startPoint.x, startPoint.y, endPoint.x, endPoint.y);
	}
}