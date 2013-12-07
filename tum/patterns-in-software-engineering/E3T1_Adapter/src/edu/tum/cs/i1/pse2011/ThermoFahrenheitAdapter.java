package edu.tum.cs.i1.pse2011;

public class ThermoFahrenheitAdapter implements ThermoInterface {

	private FahrenheitThermo fahrenheitThermo;
	
	public ThermoFahrenheitAdapter(){
		fahrenheitThermo = new FahrenheitThermo();		
	}
	
	@Override
	public double getTempC() {
		return (fahrenheitThermo.getTemp() - 32.0)*(5.0/9.0);
	}
}
