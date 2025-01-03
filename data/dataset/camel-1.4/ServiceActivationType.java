/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.camel.model;

import java.util.ArrayList;
import java.util.List;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlElementRef;
import javax.xml.bind.annotation.XmlRootElement;

import org.apache.camel.model.language.ExpressionType;

/**
 * Represents an XML &lt;serviceActivation/&gt; element
 * 
 * @version $Revision: 660266 $
 */
@XmlRootElement(name = "serviceActivation")
@XmlAccessorType(XmlAccessType.FIELD)
public class ServiceActivationType {
    @XmlAttribute
    private String group = "default";
    @XmlElementRef
    private List<ExpressionType> uris = new ArrayList<ExpressionType>();

    public String getGroup() {
        return group;
    }

    public void setGroup(String group) {
        this.group = group;
    }

    public List<ExpressionType> getUris() {
        return uris;
    }

    public void setUris(List<ExpressionType> uris) {
        this.uris = uris;
    }
}
