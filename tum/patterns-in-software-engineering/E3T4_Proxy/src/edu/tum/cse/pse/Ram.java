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

import java.util.HashMap;
import java.util.Map;

/**
 * TODO:add type description
 *
 * @author Denys Sobchyshak (denys.sobchyshak@gmail.com)
 */
public class Ram implements MemoryDevice {

	private Map<String, Integer> data;
	
	public Ram() {
		data = new HashMap<String, Integer>();
		data.put("one", 1);
		data.put("two", 2);
		data.put("three", 3);
	}
	
	/* (non-Javadoc)
	 * @see edu.tum.cse.pse.MemoryDevice#getData(java.lang.String)
	 */
	@Override
	public Integer getData(String key) {
		return data.get(key);
	}
}