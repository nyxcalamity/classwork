package edu.tum.cs.i1.pse;

import java.awt.Dimension;
import java.awt.Point;

import javax.swing.JFrame;

public class CompositeClient extends JFrame {

	private static final long serialVersionUID = -5367005543133927329L;

	public CompositeClient() {
		setSize(new Dimension(650, 650));
		setLocationRelativeTo(null);
		setResizable(false);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}

	public static void main(String[] args) {
		House home = new House();

		home.addComponent(new RectangularBlock(new Point(150, 300),150, 100)); //front rec
		home.addComponent(new RectangularBlock(new Point(200, 350),50, 50)); // door rec
		home.addComponent(new RectangularBlock(new Point(300, 300),150, 100)); //side rec
		
		home.addComponent(new Line(new Point(150, 300), new Point(225, 225))); //front left
		home.addComponent(new Line(new Point(300, 300), new Point(225, 225))); //front right
		home.addComponent(new Line(new Point(225, 225), new Point(375, 225))); //side top
		home.addComponent(new Line(new Point(375, 225), new Point(450, 300))); //side right
		home.addComponent(new RoundedWindow(new Point(375, 310), 20, 20)); //window

		Canvas component = new Canvas(home);

		CompositeClient app = new CompositeClient();
		app.getContentPane().add(component);
		app.setVisible(true);
	}
}