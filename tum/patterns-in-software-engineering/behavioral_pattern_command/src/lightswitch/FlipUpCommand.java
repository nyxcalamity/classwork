package lightswitch;

public class FlipUpCommand extends LightSwitchCommand {
	
	public FlipUpCommand(Light light) {
		this.theLight = light;
	}

	public void execute() {
		theLight.turnOn();
	}
}
