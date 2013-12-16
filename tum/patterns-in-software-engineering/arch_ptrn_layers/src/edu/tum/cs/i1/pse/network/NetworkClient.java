package edu.tum.cs.i1.pse.network;

import java.lang.reflect.InvocationTargetException;
import java.net.Socket;
import java.rmi.RemoteException;
import java.util.Random;
import java.util.concurrent.Executors;

import de.reichart.extendableserver.chatClient.ChatListener;
import de.reichart.extendableserver.chatClient.ClientConnection;
import de.reichart.extendableserver.chatClient.LoginException;
import de.reichart.extendableserver.chatClient.LoginPoint;
import de.reichart.extendableserver.chatClient.exampleGUI.ChatModule;
import de.reichart.extendableserver.common.MessageDispatcher;
import de.reichart.extendableserver.common.Messenger;
import de.reichart.extendableserver.messages.Message;
import de.reichart.extendableserver.messages.TextMessage;
import de.reichart.extendableserver.messages.toClientOnly.AddGroupMessage;
import de.reichart.extendableserver.messages.toClientOnly.AddMemberMessage;
import de.reichart.extendableserver.messages.toClientOnly.GroupListMessage;
import de.reichart.extendableserver.messages.toClientOnly.MemberListMessage;
import de.reichart.extendableserver.messages.toClientOnly.RemoveGroupMessage;
import de.reichart.extendableserver.messages.toClientOnly.RemoveMemberMessage;
import de.reichart.extendableserver.messages.toClientOnly.ServerDisconnectMessage;
import edu.tum.cs.i1.pse.application.ApplicationLayerInterface;
import edu.tum.cs.i1.pse.presentation.PresentationLayerInterface;

public class NetworkClient implements NetworkLayerInterface, Messenger, LoginPoint, ChatListener {
	private PresentationLayerInterface presentationLayer;
	private ApplicationLayerInterface applicationLayer;

	private final MessageDispatcher dispatcher;
	private final ClientConnection connection;
	
	public NetworkClient(String host, int port, LoginPoint loginPoint) {
		dispatcher = new MessageDispatcher();
		try {
			connection = new ClientConnection(new Socket(host, port), dispatcher, this);
			Executors.newSingleThreadExecutor().submit(connection);
		} catch (Exception e) {
			System.out.println("Could not establish connection with host");
			throw new RuntimeException("Can not resolve the hosts IP address", e);
		}
		ChatModule chatModule = new ChatModule();
		chatModule.setMessenger(this);
		dispatcher.addModule(chatModule);
		chatModule.addChatListener(this);
		
		try {
			connection.attemptLogin();
		} catch (LoginException e) {
			e.printStackTrace();
		}
	}
	
	@Override
	public void sendMessage(String message) {
		sendMessage(new TextMessage(message));
	}

	@Override
	public void receiveMessage(String message) {
		if (presentationLayer != null) {
			presentationLayer.receiveMessage(message);
		} else if (applicationLayer != null) {
			applicationLayer.receiveMessage(message);
		}
	}

	@Override
	public void handleMessage(Message message) throws RemoteException,
			IllegalArgumentException, IllegalAccessException,
			InvocationTargetException {
		if (message instanceof TextMessage) {
			TextMessage textMessage = (TextMessage) message;
			receiveMessage(textMessage.getText());
		}
	}

	@Override
	public void sendMessage(Message message) {
		connection.sendMessage(message);
	}

	@Override
	public void sendPrivateMessage(String name, Message message) {
		// empty
	}

	@Override
	public String askForUserName(String message) {
		Random rand = new Random(System.currentTimeMillis());
		int userNumber = rand.nextInt();
		String username = new String("user" + userNumber);
		return username;
	}

	@Override
	public void loginSuccess(String username) throws LoginException {
		System.out.println("Logged in successfully");
	}

	@Override
	public void handleTextMessage(TextMessage message) {
		if (!message.getSender().equals(connection.getUser().getUsername())) {
			receiveMessage(message.getText());
		}
	}

	@Override
	public void handleRemoveGroupMessage(RemoveGroupMessage message) {
		// empty
	}

	@Override
	public void handleGroupListMessage(GroupListMessage message) {
		// empty
	}

	@Override
	public void handleMemberListMessage(MemberListMessage message) {
		// empty
	}

	@Override
	public void handleAddMemberMessage(AddMemberMessage message) {
		// empty
	}

	@Override
	public void handleRemoveMemberMessage(RemoveMemberMessage message) {
		// empty
	}

	@Override
	public void handleAddGroupMessage(AddGroupMessage message) {
		// empty
	}

	@Override
	public void handleServerDisconnectMessage(ServerDisconnectMessage message) {
		// empty
	}

	@Override
	public void setPresentationLayer(
			PresentationLayerInterface presentationLayer) {
		this.presentationLayer = presentationLayer;
	}

	@Override
	public void setApplicationLayer(ApplicationLayerInterface applicationLayer) {
		this.applicationLayer = applicationLayer;
	}
	
}
