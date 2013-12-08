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

import java.awt.Graphics;
import java.util.ArrayList;
import java.util.List;

/**
 * TODO:add type description
 *
 * @author Denys Sobchyshak (denys.sobchyshak@gmail.com)
 */
public class AbstractCompositeComponent extends AbstractComponent {
	
	protected List<AbstractComponent> children;
	
	public AbstractCompositeComponent() {
		children = new ArrayList<AbstractComponent>();
	}

	/* (non-Javadoc)
	 * @see edu.tum.cs.i1.pse.AbstractComponent#draw(java.awt.Graphics)
	 */
	@Override
	public void draw(Graphics graphics) {
		for (AbstractComponent child : children)
			child.draw(graphics);
	}
	
	public void addComponent(AbstractComponent component){
		if (component != null)
			children.add(component);
	}
	
	public void removeComponent(AbstractComponent component){
		if (component != null)
			children.remove(component);
	}
	
	public List<AbstractComponent> getChildren(){
		return children;
	}
}