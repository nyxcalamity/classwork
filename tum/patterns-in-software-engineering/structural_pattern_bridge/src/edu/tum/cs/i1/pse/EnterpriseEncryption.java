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

/**
 * Represents encryption facade for enterprise use.
 *
 * @author Denys Sobchyshak (denys.sobchyshak@gmail.com)
 */
public class EnterpriseEncryption extends Encryption {

	public EnterpriseEncryption() {
		cipher = new Caesar();
	}
	
	/* (non-Javadoc)
	 * @see edu.tum.cs.i1.pse.Encryption#encrypt(java.lang.String, byte)
	 */
	@Override
	public String encrypt(String string, byte key) {
		if (key < 10)
			throw new IllegalArgumentException("Key should be at least as large as 10. Provided key: " + key);
		return super.encrypt(string, key);
	}

	/* (non-Javadoc)
	 * @see edu.tum.cs.i1.pse.Encryption#decrypt(java.lang.String, byte)
	 */
	@Override
	public String decrypt(String string, byte key) {
		if (key < 10)
			throw new IllegalArgumentException("Key should be at least as large as 10. Provided key: " + key);
		return super.decrypt(string, key);
	}
}