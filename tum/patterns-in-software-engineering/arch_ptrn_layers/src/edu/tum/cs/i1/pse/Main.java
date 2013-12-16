package edu.tum.cs.i1.pse;

import java.security.NoSuchAlgorithmException;

import javax.crypto.KeyGenerator;

import edu.tum.cs.i1.pse.application.ApplicationLayerInterface;
import edu.tum.cs.i1.pse.application.CliClient;
import edu.tum.cs.i1.pse.network.NetworkClient;
import edu.tum.cs.i1.pse.network.NetworkLayerInterface;
import edu.tum.cs.i1.pse.presentation.CaesarEncryption;
import edu.tum.cs.i1.pse.presentation.PresentationLayerInterface;

public class Main {
	
	public static void main(String[] args) {
		String hostname = "ec2-54-217-104-219.eu-west-1.compute.amazonaws.com";
		int port = 1337;
		
		NetworkLayerInterface networkLayer = new NetworkClient(hostname, port, null);
		PresentationLayerInterface presentationLayer = new CaesarEncryption(11);
		
		networkLayer.setPresentationLayer(presentationLayer);
		presentationLayer.setNetworkLayer(networkLayer);
		
		//TODO: Instantiate your chat client here
		ApplicationLayerInterface applicationLayer = new CliClient();
		networkLayer.setApplicationLayer(applicationLayer);
		presentationLayer.setApplicationLayer(applicationLayer);
		applicationLayer.setNetworkLayer(networkLayer);
		applicationLayer.setPresentationLayer(presentationLayer);
		applicationLayer.sendMessage("WHATEVER!");
	}
	
	static byte[] getFixedAESKey() {
		return new byte[]{42, 13, 12, -94, 2, -91, 78, -121, 76, -77, 119, 122, -32, -67, 6, -43};
	}
	
	static byte[] generateAESKey() {
		KeyGenerator kgen;
		try {
			kgen = KeyGenerator.getInstance("AES");
		} catch (NoSuchAlgorithmException e) {
			throw new RuntimeException(e);
		}
		kgen.init(128);
		byte[] key = kgen.generateKey().getEncoded();
		return key;
	}

}