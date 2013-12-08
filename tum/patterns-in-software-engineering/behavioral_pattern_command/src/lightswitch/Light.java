package lightswitch;

import java.awt.Label;

public class Light extends Label{

	private static final long serialVersionUID = -4640177502127407955L;

	public void turnOn() {
		System.out.println("The light is on");
	}

	public void turnOff() {
		System.out.println("The light is off");
	}
}
