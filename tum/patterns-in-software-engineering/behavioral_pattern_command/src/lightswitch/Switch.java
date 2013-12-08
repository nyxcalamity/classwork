package lightswitch;

public class Switch {
	private LightSwitchCommand flipUpCommand;
	private LightSwitchCommand flipDownCommand;

	public Switch(LightSwitchCommand flipUpCmd, LightSwitchCommand flipDownCmd) {
		this.flipUpCommand = flipUpCmd;
		this.flipDownCommand = flipDownCmd;
	}

	public void flipUp() {
		flipUpCommand.execute();
	}

	public void flipDown() {
		flipDownCommand.execute();
	}
}
