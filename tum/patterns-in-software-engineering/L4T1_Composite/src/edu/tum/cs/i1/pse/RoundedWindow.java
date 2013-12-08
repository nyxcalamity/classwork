package edu.tum.cs.i1.pse;

import java.awt.Point;
import java.util.ArrayList;
import java.util.List;

import edu.tum.cs.i1.pse.exc.NegativeValueException;

public class RoundedWindow {
	private Point top;
	private int width;
	private int height;
	private ArrayList<Line> lineList = new ArrayList<Line>();
	private Circle circle;

	public Circle getCircle() {
		return circle;
	}

	public void setCircle(Circle circle) {
		this.circle = circle;
	}

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
		this.circle = new Circle(top, width, height);
		Line a = new Line(new Point(top.x + width / 2, top.y), new Point(top.x
				+ width / 2, top.y + height));
		Line b = new Line(new Point(top.x, top.y + height / 2), new Point(top.x
				+ width, top.y + height / 2));

		lineList.add(a);
		lineList.add(b);
	}

	public List<Line> getLineList() {
		return lineList;
	}
}