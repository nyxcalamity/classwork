package edu.tum.cs.i1.pse.presentation;

import edu.tum.cs.i1.pse.application.ApplicationLayerInterface;
import edu.tum.cs.i1.pse.network.NetworkLayerInterface;

public interface PresentationLayerInterface {

	void sendMessage(String message);
	void receiveMessage(String message);
	
	void setNetworkLayer(NetworkLayerInterface networkLayer);
	void setApplicationLayer(ApplicationLayerInterface applicationLayer);	
}
