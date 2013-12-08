package edu.tum.cs.i1.pse;

import java.awt.Graphics;
import java.util.ArrayList;

public class House {

	private ArrayList<RectangularBlock> rectangleList = new ArrayList<RectangularBlock>();
	private ArrayList<Line> lineList = new ArrayList<Line>();
	private ArrayList<RoundedWindow> windowList = new ArrayList<RoundedWindow>();

	public House() {
	}

	public void addRectangle(RectangularBlock shape) {
		if (shape != null) {
			rectangleList.add(shape);
		}
	}

	public void addLine(Line shape) {
		if (shape != null) {
			lineList.add(shape);
		}
	}

	public void addCircle(RoundedWindow shape) {
		if (shape != null) {
			windowList.add(shape);
		}
	}

	public void makeHouse(Graphics g) {
		for (RoundedWindow item : windowList) {
			item.getCircle().actualDraw(g);
			for (Line lineItem : item.getLineList()) {
				lineItem.actualDraw(g);
			}
		}
		for (RectangularBlock item : rectangleList) {
			for (Line lineItem : item.getLineList())
				lineItem.actualDraw(g);
		}
		for (Line lineItem : lineList) {
			lineItem.actualDraw(g);
		}

	}

}