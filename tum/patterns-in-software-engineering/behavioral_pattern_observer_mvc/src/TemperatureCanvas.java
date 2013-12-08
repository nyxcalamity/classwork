

import java.awt.Canvas;
import java.awt.Color;
import java.awt.Graphics;

public class TemperatureCanvas extends Canvas {
	
	private static final long serialVersionUID = -450878788391225566L;
	private TemperatureGauge gauge;
	private Color fillcolor;
	private static final int width = 20;
	private static final int top = 20;
	private static final int height = 200;
	private static final int left = 100;
	@SuppressWarnings("unused")
	private static final int right = 250;
	
	public TemperatureCanvas(TemperatureGauge g) {
		gauge = g;
	}

	public void paint(Graphics g) {
		g.setColor(Color.black);
		g.drawRect(left, top, width, height);
		if(gauge.get() > 0.0)
			fillcolor = Color.red;
		else
			fillcolor = Color.blue;
		g.setColor(fillcolor);
		g.fillOval(left - width / 2, top + height - width / 3, width * 2,
				width * 2);
		g.setColor(Color.black);
		g.drawOval(left - width / 2, top + height - width / 3, width * 2,
				width * 2);
		g.setColor(Color.white);
		g.fillRect(left + 1, top + 1, width - 1, height - 1);
		g.setColor(fillcolor);
		long redtop = height * (gauge.get() - gauge.getMax())
				/ (gauge.getMin() - gauge.getMax());
		g.fillRect(left + 1, top + (int) redtop, width - 1, height
				- (int) redtop);
	}
}
