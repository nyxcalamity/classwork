package edu.tum.cs.i1.pse.presentation;

import edu.tum.cs.i1.pse.application.ApplicationLayerInterface;
import edu.tum.cs.i1.pse.network.NetworkLayerInterface;

public class CaesarEncryption implements PresentationLayerInterface {
	private ApplicationLayerInterface applicationLayer;
	private NetworkLayerInterface networkLayer;
	private final int key;
	
	public CaesarEncryption(int key) {
		if (key <= 0 || key >= 26) {
			throw new IllegalArgumentException("The key must have a value between 1 to 25");
		}
		this.key = key;
	}

	public String encrypt(String w, int k) { 		
		String result = new String();
		for (int i = 0; i < w.length(); i++) {
			char ch = w.charAt(i);
			if (ch >= 'A' && ch <= 'Z') {
				ch = shift('A', ch, key);
			} else if (ch >= 'a' && ch <= 'z') {
				ch = shift('a', ch, key);
			}
			result = result + ch;
		}
		return result;
	}

	private char shift(char offset, char input, int key) {
		return (char)(offset + (input - offset + key) % 26);
	}
	
	public String decrypt(String w, int k) {
		String result = new String();
		for (int i = 0; i < w.length(); i++) {
			char ch = w.charAt(i);
			if (ch >= 'A' && ch <= 'Z') {
				ch = shift('A', ch, 26 - key);
			} else if (ch >= 'a' && ch <= 'z') {
				ch = shift('a', ch, 26 - key);
			}
			result = result  + ch;
		}
		return result; 
	}

	@Override
	public void sendMessage(String message) {
		String encryptedMessage = encrypt(message, key);
		System.out.println("Sent encrypted message " + encryptedMessage);
		networkLayer.sendMessage(encryptedMessage);
	}

	@Override
	public void receiveMessage(String message) {
		String decryptedMessage = decrypt(message, key);
		System.out.println("Received decrypted message: " + decryptedMessage);
		applicationLayer.receiveMessage(decryptedMessage);
	}

	@Override
	public void setApplicationLayer(ApplicationLayerInterface applicationLayer) {
		this.applicationLayer = applicationLayer;
	}

	@Override
	public void setNetworkLayer(NetworkLayerInterface networkLayer) {
		this.networkLayer = networkLayer;
	}

}
