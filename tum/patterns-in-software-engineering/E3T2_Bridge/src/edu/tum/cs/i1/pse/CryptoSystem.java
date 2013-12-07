package edu.tum.cs.i1.pse;

public class CryptoSystem {

	private Encryption encryption;
	
	public CryptoSystem(String encryptionType) {
		if (encryptionType.equalsIgnoreCase("Enterprise"))
			encryption = new EnterpriseEncryption();
	    else 
	    	encryption = new PersonalEncryption();	
	}

	public String encryptDoc(String plain, byte key) {
		return encryption.encrypt(plain, key);
	}

	public String decryptDoc(String secret, byte key) {
		return encryption.decrypt(secret, key);
	}
}