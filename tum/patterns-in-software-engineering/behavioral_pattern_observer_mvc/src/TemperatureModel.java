public class TemperatureModel extends java.util.Observable {

	private double temperatureC = 0.0;
	
	public double getF() {
		return (temperatureC * 9.0 / 5.0) + 32.0;
	}

	public double getC() {
		return temperatureC;
	}
	
	public double getK(){
		return temperatureC + 273.15;
	}

	public void setF(double tempF) {
		temperatureC = (tempF - 32.0) * 5.0 / 9.0;
		setChanged();
		notifyObservers();
	}

	public void setC(double tempC) {
		temperatureC = tempC;
		setChanged();
		notifyObservers();
	}
	
	public void setK(double tempInKelvin) {
		temperatureC = tempInKelvin - 273.15;
		setChanged();
		notifyObservers();
	}
}
