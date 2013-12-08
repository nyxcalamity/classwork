 

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Observable;

public class CelsiusGUI extends TemperatureGUI {

	public CelsiusGUI(TemperatureModel model, int h, int v) {
		super("Celsius Temperature", model, h, v);
		setDisplay("" + model.getC());
		addRaiseTempListener(new RaiseTempListener());
		addLowerTempListener(new LowerTempListener());
		addDisplayListener(new DisplayListener());
	}

	public void update(Observable t, Object o) { // Called from the Model
		setDisplay("" + model().getC());
	}

	class RaiseTempListener implements ActionListener {
		public void actionPerformed(ActionEvent e) {
			model().setC(model().getC() + 1.0);
		}
	}

	class LowerTempListener implements ActionListener {
		public void actionPerformed(ActionEvent e) {
			model().setC(model().getC() - 1.0);
		}
	}

	class DisplayListener implements ActionListener {
		public void actionPerformed(ActionEvent e) {
			double value = getDisplay();
			model().setC(value);
		}
	}
}