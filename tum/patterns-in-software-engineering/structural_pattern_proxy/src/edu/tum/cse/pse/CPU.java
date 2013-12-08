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
package edu.tum.cse.pse;

/**
 * TODO:add type description
 *
 * @author Denys Sobchyshak (denys.sobchyshak@gmail.com)
 */
public class CPU {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		MemoryDevice device = new CacheProxy();
		requestData(device, "one");
		requestData(device, "two");
		requestData(device, "three");
	}
	
	public static void requestData(MemoryDevice device, String key){
		System.out.println("Requesting from memory: " + key);
		System.out.println("Obtained value: " + device.getData(key).toString());
	}
}