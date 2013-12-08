package lightswitch;

public class FlipDownCommand extends LightSwitchCommand {
	
	public FlipDownCommand(Light light) {
		this.theLight = light;
	}

	public void execute() {
		theLight.turnOff();
	}
}
