/*
 * Copyright 2001-2004 The Apache Software Foundation.
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
/*
 * $Id$
 */

package org.apache.xalan.xsltc.compiler;

import org.apache.xalan.xsltc.compiler.util.Util;

/**
 * @author Jacek Ambroziak
 * @author Santiago Pericas-Geertsen
 */
final class Attribute extends Instruction {
    private QName _name;
	
    public void display(int indent) {
	indent(indent);
	Util.println("Attribute " + _name);
	displayContents(indent + IndentIncrement);
    }

    public void parseContents(Parser parser) {
	_name = parser.getQName(getAttribute("name"));
	parseChildren(parser);
	//!!! add text nodes
	//!!! take care of value templates
    }
}
