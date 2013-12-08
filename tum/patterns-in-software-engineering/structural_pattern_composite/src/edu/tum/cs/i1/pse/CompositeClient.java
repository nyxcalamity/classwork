package edu.tum.cs.i1.pse;

import java.awt.Dimension;
import java.awt.Point;

import javax.swing.JFrame;

public class CompositeClient extends JFrame {

	private static final long serialVersionUID = -5367005543133927329L;

	public CompositeClient() {
		super();
		setSize(new Dimension(650, 650));
		setLocationRelativeTo(null);
		setResizable(false);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}

	public static void main(String[] args) {
		House home = new House();
		RectangularBlock frontRec = new RectangularBlock(new Point(150, 300),
				150, 100);
		RectangularBlock doorRec = new RectangularBlock(new Point(200, 350),
				50, 50);
		RectangularBlock sideRec = new RectangularBlock(new Point(300, 300),
				150, 100);

		Line frontLeft = new Line(new Point(150, 300), new Point(225, 225));
		Line fromtRight = new Line(new Point(300, 300), new Point(225, 225));
		Line sideTop = new Line(new Point(225, 225), new Point(375, 225));
		Line sideRight = new Line(new Point(375, 225), new Point(450, 300));

		RoundedWindow windowRounded = new RoundedWindow(new Point(375, 310),
				20, 20);

		home.addRectangle(frontRec);
		home.addRectangle(doorRec);
		home.addRectangle(sideRec);
		home.addLine(frontLeft);
		home.addLine(fromtRight);
		home.addLine(sideTop);
		home.addLine(sideRight);
		home.addCircle(windowRounded);

		Canvas component = new Canvas(home);

		CompositeClient app = new CompositeClient();
		app.getContentPane().add(component);
		app.setVisible(true);
	}
}
