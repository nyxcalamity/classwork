package edu.tum.cs.i1.pse;

import java.awt.Graphics;
import java.awt.Point;
import edu.tum.cs.i1.pse.exc.NegativeValueException;

public class Circle {
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

	public void actualDraw(Graphics g) {
		g.drawOval(top.x, top.y, width, height);
	}
}