package edu.tum.cs.i1.pse;

import java.awt.Graphics;
import java.awt.Point;

import edu.tum.cs.i1.pse.exc.NegativeValueException;

public class Line {
	private Point startPoint, endPoint;

	public Line(Point startPoint, Point endPoint) {
		if (startPoint.x < 0 || startPoint.y < 0 || endPoint.x < 0
				|| endPoint.y < 0) {
			throw new NegativeValueException();
		} else {
			this.startPoint = startPoint;
			this.endPoint = endPoint;
		}
	}

	public void actualDraw(Graphics g) {
		g.drawLine(startPoint.x, startPoint.y, endPoint.x, endPoint.y);
	}

}