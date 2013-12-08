package edu.tum.cs.i1.pse;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;

import javax.swing.JComponent;

public class Canvas extends JComponent {

	private static final long serialVersionUID = 6954199689687046200L;
	private House shape;

	public Canvas(House single) {
		super();
		this.shape = single;
	}

	@Override
	protected void paintComponent(Graphics g) {
		super.paintComponent(g);
		Graphics2D g2d = (Graphics2D) g;
		g2d.setColor(Color.WHITE);
		g2d.fillRect(0, 0, 700, 700);

		g2d.setStroke(new BasicStroke(2));
		g2d.setColor(Color.BLACK);
		shape.draw(g2d);
		g2d.setColor(Color.RED);
	}
}