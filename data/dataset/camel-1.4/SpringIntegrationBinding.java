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
package org.apache.camel.component.spring.integration;

import org.apache.camel.Exchange;
import org.springframework.integration.message.GenericMessage;


/**
 * The helper class for Mapping between the Spring Integration message and
 * the Camel Message
 * @version $Revision: 652240 $
 */
public final class SpringIntegrationBinding {

    private SpringIntegrationBinding() {
        // Helper class
    }

    @SuppressWarnings("unchecked")
    public static org.springframework.integration.message.Message createSpringIntegrationMessage(Exchange exchange) {
        org.apache.camel.Message message = exchange.getIn();
        GenericMessage siMessage = new GenericMessage(message.getBody());
        return siMessage;
    }

    @SuppressWarnings("unchecked")
    public static org.springframework.integration.message.Message storeToSpringIntegrationMessage(org.apache.camel.Message message) {
        GenericMessage siMessage = new GenericMessage(message.getBody());
        return siMessage;
    }

    public static void storeToCamelMessage(org.springframework.integration.message.Message siMessage, org.apache.camel.Message cMessage) {
        cMessage.setBody(siMessage.getPayload());
        //TODO copy the message header
    }

}
