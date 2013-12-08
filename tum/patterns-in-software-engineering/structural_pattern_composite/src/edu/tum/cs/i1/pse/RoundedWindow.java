package edu.tum.cs.i1.pse;

import java.awt.Graphics;
import java.awt.Point;

import edu.tum.cs.i1.pse.exc.NegativeValueException;

public class RoundedWindow extends AbstractCompositeComponent {
	private Point top;
	private int width;
	private int height;
	private Circle circle;

	public RoundedWindow(Point top, int width, int hegiht) {
		if (top.x < 0 || top.y < 0 || width < 0 || height < 0) {
			throw new NegativeValueException();
		} else {
			this.top = top;
			this.width = width;
			this.height = hegiht;
			makeCircularWindow();
		}
	}

	private void makeCircularWindow() {
		circle = new Circle(top, width, height);
		addComponent(new Line(new Point(top.x+width/2, top.y), new Point(top.x+width/2, top.y+height)));
		addComponent(new Line(new Point(top.x, top.y+height/2), new Point(top.x+width, top.y+height/2)));
	}
	
	/* (non-Javadoc)
	 * @see edu.tum.cs.i1.pse.AbstractCompositeComponent#draw(java.awt.Graphics)
	 */
	@Override
	public void draw(Graphics graphics) {
		super.draw(graphics);
		circle.draw(graphics);
	}

	public Circle getCircle() {
		return circle;
	}

	public void setCircle(Circle circle) {
		this.circle = circle;
	}
}