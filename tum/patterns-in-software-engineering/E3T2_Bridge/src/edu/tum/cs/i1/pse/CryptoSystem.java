package edu.tum.cs.i1.pse;

public class CryptoSystem {

	private Cipher imp;
	
	public CryptoSystem(String encryptionType) {
		if (encryptionType.equalsIgnoreCase("Enterprise"))
			imp = new Caesar();
	    else 
	    	imp = new Transpose();	
	}

	public String encryptDoc(String plain, byte key) {
		return imp.encryptWord(plain, key);
	}

	public String decryptDoc(String secret, byte key) {
		return imp.decryptWord(secret, key);
	}

}
