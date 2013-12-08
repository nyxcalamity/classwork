package edu.tum.cs.i1.pse;

import java.awt.Point;

import edu.tum.cs.i1.pse.exc.NegativeValueException;

public class RectangularBlock extends AbstractCompositeComponent {
	private Point topLeft;
	private Point topRight;
	private Point bottomLeft;
	private Point bottomRight;

	public RectangularBlock(Point startPoint, int width, int height) {
		if (startPoint.x < 0 || startPoint.y < 0 || width < 0 || height < 0) {
			throw new NegativeValueException();
		} else {
			this.topLeft = startPoint;
			this.topRight = new Point((topLeft.x + width), topLeft.y);
			this.bottomLeft = new Point(topLeft.x, (topLeft.y + height));
			this.bottomRight = new Point(topLeft.x + width, topLeft.y + height);
			makeRactangle();
		}
	}

	private void makeRactangle() {
		addComponent(new Line(topLeft, topRight));
		addComponent(new Line(topLeft, bottomLeft));
		addComponent(new Line(bottomLeft, bottomRight));
		addComponent(new Line(topRight, bottomRight));
	}
}