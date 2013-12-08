package edu.tum.cs.i1.pse;

import java.awt.Point;
import java.util.ArrayList;
import java.util.List;

import edu.tum.cs.i1.pse.exc.NegativeValueException;

public class RectangularBlock {
	private Point topLeft;
	private Point topRight;
	private Point bottomLeft;
	private Point bottomRight;
	private ArrayList<Line> lineList = new ArrayList<Line>();

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
		Line a = new Line(topLeft, topRight);
		Line b = new Line(topLeft, bottomLeft);
		Line c = new Line(bottomLeft, bottomRight);
		Line d = new Line(topRight, bottomRight);
		lineList.add(a);
		lineList.add(b);
		lineList.add(c);
		lineList.add(d);
	}

	public List<Line> getLineList() {
		return lineList;
	}
}