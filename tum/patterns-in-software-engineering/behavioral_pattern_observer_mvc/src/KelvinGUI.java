/**
 * Copyright 2013 Denys Sobchyshak
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Observable;


/**
 * TODO:add type description
 *
 * @author Denys Sobchyshak (denys.sobchyshak@gmail.com)
 */
public class KelvinGUI extends TemperatureGUI {

	/**
	 * @param label
	 * @param model
	 * @param h
	 * @param v
	 */
	KelvinGUI(String label, TemperatureModel model, int h, int v) {
		super(label, model, h, v);
		setDisplay("" + model.getK());
		addRaiseTempListener(new RaiseTempListener());
		addLowerTempListener(new LowerTempListener());
		addDisplayListener(new DisplayListener());
	}

	/* (non-Javadoc)
	 * @see java.util.Observer#update(java.util.Observable, java.lang.Object)
	 */
	@Override
	public void update(Observable o, Object arg) {
		setDisplay("" + model().getK());
	}

	class RaiseTempListener implements ActionListener {
		public void actionPerformed(ActionEvent e) {
			model().setK(model().getK() + 1.0);
		}
	}

	class LowerTempListener implements ActionListener {
		public void actionPerformed(ActionEvent e) {
			model().setK(model().getK() - 1.0);
		}
	}

	class DisplayListener implements ActionListener {
		public void actionPerformed(ActionEvent e) {
			double value = getDisplay();
			model().setK(value);
		}
	}
}
