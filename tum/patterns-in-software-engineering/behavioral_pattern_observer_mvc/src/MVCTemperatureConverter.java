

public class MVCTemperatureConverter {
	
	public static void main(String args[]) {
		TemperatureModel temperature = new TemperatureModel();
		new FahrenheitGUI(temperature, 100, 100);
		new KelvinGUI("Kelvin Temperature", temperature, 100, 400);
		new CelsiusGUI(temperature, 100, 250);
		new SliderGUI(temperature, 100, 20);
		new GraphGUI(temperature, 300, 200);
	}
}
