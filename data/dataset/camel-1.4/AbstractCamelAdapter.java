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

package org.apache.camel.component.spring.integration.adapter;

import org.apache.camel.CamelContext;
import org.apache.camel.Consumer;
import org.apache.camel.Endpoint;
import org.springframework.integration.handler.MessageHandler;

/**
 * The Abstract class for the Spring Integration Camel Adapter
 *
 * @author Willem Jiang
 *
 * @version $Revision: 655008 $
 */
public abstract class AbstractCamelAdapter implements MessageHandler {
    private CamelContext camelContext;
    private String camelEndpointUri;
    private volatile boolean expectReply = true;

    public void setCamelContext(CamelContext context) {
        camelContext = context;
    }

    public CamelContext getCamelContext() {
        return camelContext;
    }

    public String getCamelEndpointUri() {
        return camelEndpointUri;
    }

    public void setCamelEndpointUri(String uri) {
        camelEndpointUri = uri;
    }

    public void setExpectReply(boolean expectReply) {
        this.expectReply = expectReply;
    }

    public boolean isExpectReply() {
        return expectReply;
    }


}
