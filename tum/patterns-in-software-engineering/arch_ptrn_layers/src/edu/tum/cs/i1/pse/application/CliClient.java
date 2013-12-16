/**
 * Copyright 2013 Denys Sobchyshak
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package edu.tum.cs.i1.pse.application;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import edu.tum.cs.i1.pse.network.NetworkLayerInterface;
import edu.tum.cs.i1.pse.presentation.PresentationLayerInterface;

/**
 * TODO:add type description
 *
 * @author Denys Sobchyshak (denys.sobchyshak@gmail.com)
 */
public class CliClient implements ApplicationLayerInterface {
	
	private PresentationLayerInterface presentationInterface;
	private NetworkLayerInterface networkInterface;
	
	/* (non-Javadoc)
	 * @see edu.tum.cs.i1.pse.application.ApplicationLayerInterface#sendMessage(java.lang.String)
	 */
	@Override
	public void sendMessage(String message) {
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		String name = "";
		try {
			System.out.print("Please enter your name: \t");
			name = br.readLine();
		} catch (IOException ex){
			System.out.println("Could not read line.");
		} finally {
			if (presentationInterface != null)
				presentationInterface.sendMessage("Hello all, from " + name);
			if (networkInterface != null)
				networkInterface.sendMessage("Hello all, from " + name);
		}
	}

	/* (non-Javadoc)
	 * @see edu.tum.cs.i1.pse.application.ApplicationLayerInterface#receiveMessage(java.lang.String)
	 */
	@Override
	public void receiveMessage(String message) {
		System.out.println("Received: " + message);
	}

	/* (non-Javadoc)
	 * @see edu.tum.cs.i1.pse.application.ApplicationLayerInterface#setPresentationLayer(edu.tum.cs.i1.pse.presentation.PresentationLayerInterface)
	 */
	@Override
	public void setPresentationLayer(PresentationLayerInterface presentationLayer) {
		this.presentationInterface = presentationLayer;
	}

	/* (non-Javadoc)
	 * @see edu.tum.cs.i1.pse.application.ApplicationLayerInterface#setNetworkLayer(edu.tum.cs.i1.pse.network.NetworkLayerInterface)
	 */
	@Override
	public void setNetworkLayer(NetworkLayerInterface networkLayer) {
		this.networkInterface = networkLayer;
	}
}
