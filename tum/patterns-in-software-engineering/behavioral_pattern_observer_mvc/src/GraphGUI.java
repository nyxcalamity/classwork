

import java.awt.Canvas;
import java.awt.Frame;
import java.awt.Graphics;
import java.awt.Panel;
import java.util.Observable;
import java.util.Observer;

public class GraphGUI extends Frame implements Observer {

	private static final long serialVersionUID = 4385762751045760627L;
	private TemperatureModel model;
	private Canvas gaugeCanvas;
	private TemperatureGauge gauge;
	
	public GraphGUI(TemperatureModel model, int h, int v) {
		super("Temperature Gauge");
		this.model = model;
		gauge = new TemperatureGauge(-20, 150);
		Panel Top = new Panel();
		add("North", Top);
		gaugeCanvas = new TemperatureCanvas(gauge);
		gaugeCanvas.setSize(500, 280);
		add("Center", gaugeCanvas);
		setSize(220, 300);
		setLocation(h, v);
		setVisible(true);
		model.addObserver(this); // Connect to the model
	}

	public void update(Observable obs, Object o) { // Respond to changes
		repaint();
	}

	public void paint(Graphics g) {
		int celsius = (int) model.getC(); // Use the current data to paint
		gauge.set(celsius);
		gaugeCanvas.repaint();
		super.paint(g);
	}
}
