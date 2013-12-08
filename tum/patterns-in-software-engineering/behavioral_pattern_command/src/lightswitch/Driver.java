package lightswitch;

import java.applet.Applet;
import java.awt.Button;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;


public class Driver extends Applet {

	private static final long serialVersionUID = -1772092802255949754L;
	
	Button switchUpButton;
	Button switchDownButton;
	Light lamp;
	
	LightSwitchCommand switchUp;
	LightSwitchCommand switchDown;
	Switch lightSwitch;
	
	@Override
	public void init() {
		setLayout(new GridLayout(3,2));
		switchUpButton = new Button("switch up"); 
		switchDownButton = new Button("switch down"); 
		lamp = new Light();
		lamp.setText("Lamp here");
		
		switchDownButton.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				lightSwitch.flipDown();
			}
		});
		switchUpButton.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				lightSwitch.flipUp();
			}
		});
		
		add(switchUpButton);
		add(switchDownButton);
		add(lamp);
		
		lamp = new Light();
		switchUp = new FlipUpCommand(lamp);
		switchDown = new FlipDownCommand(lamp);
		lightSwitch = new Switch(switchUp, switchDown);
		
	}
	
	public static void main(String[] args) {
		
		Driver driver = new Driver();
		driver.init();
		
		try {
			if (args[0].equalsIgnoreCase("ON"))
				driver.lightSwitch.flipUp();
			else if (args[0].equalsIgnoreCase("OFF"))
				driver.lightSwitch.flipDown();
		} catch (Exception e) {
			System.out.println("Argument ON or OFF required.");
		}
	}
}
