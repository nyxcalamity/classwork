package edu.tum.cs.i1.pse.exc;

public class NegativeValueException extends ArithmeticException {

	private static final long serialVersionUID = 1L;

	public NegativeValueException() {
		super("Error, Negative values not allowed!");
	}

}
