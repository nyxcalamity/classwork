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
package edu.tum.cs.i1.pse;

import java.util.StringTokenizer;

/**
 * Abstract encryption type.
 *
 * @author Denys Sobchyshak (denys.sobchyshak@gmail.com)
 */
public abstract class Encryption {
	
	protected Cipher cipher;
	
	public String encrypt(String string, byte key){
		String encryptedString = "";
		StringTokenizer st = new StringTokenizer(string);
	    while (st.hasMoreTokens())
	    	encryptedString += "" + cipher.encryptWord(st.nextToken(), key); 
		return encryptedString;
	}
	public String decrypt(String string, byte key){
		String decryptedSring = "";
		StringTokenizer st = new StringTokenizer(string);
	    while (st.hasMoreTokens())
	    	decryptedSring += "" + cipher.decryptWord(st.nextToken(), key); 
		return decryptedSring;
	}
}
