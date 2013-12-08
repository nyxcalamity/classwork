

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Observable;

public class FahrenheitGUI extends TemperatureGUI {
	
	public FahrenheitGUI(TemperatureModel model, int h, int v) {
		super("Fahrenheit Temperature", model, h, v);
		setDisplay("" + model.getF());
		addRaiseTempListener(new RaiseTempListener());
		addLowerTempListener(new LowerTempListener());
		addDisplayListener(new DisplayListener());
	}

	public void update(Observable t, Object o) { // Called from the Model
		setDisplay("" + model().getF());
	}

	class RaiseTempListener implements ActionListener {
		public void actionPerformed(ActionEvent e) {
			model().setF(model().getF() + 1.0);
		}
	}

	class LowerTempListener implements ActionListener {
		public void actionPerformed(ActionEvent e) {
			model().setF(model().getF() - 1.0);
		}
	}

	class DisplayListener implements ActionListener {
		public void actionPerformed(ActionEvent e) {
			double value = getDisplay();
			model().setF(value);
		}
	}
}
