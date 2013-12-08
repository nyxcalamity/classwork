

public class TemperatureGauge {
	private int max, min, current;
	
	public TemperatureGauge(int min, int max) {
		this.min = min;
		this.max = max;
	}

	public void set(int level) {
		current = level;
	}

	public int get() {
		return current;
	}

	public int getMax() {
		return max;
	}

	public int getMin() {
		return min;
	}
}
