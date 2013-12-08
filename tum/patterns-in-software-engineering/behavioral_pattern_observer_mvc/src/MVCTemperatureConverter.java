

public class MVCTemperatureConverter {
	
	public static void main(String args[]) {
		TemperatureModel temperature = new TemperatureModel();
		new FahrenheitGUI(temperature, 100, 100);
		new CelsiusGUI(temperature, 100, 250);
		new SliderGUI(temperature, 20, 100);
		new GraphGUI(temperature, 200, 200);
	}
}
