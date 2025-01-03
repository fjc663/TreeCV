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
package org.apache.camel.component.xmpp;

import org.apache.camel.Exchange;
import org.apache.camel.impl.DefaultProducer;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.jivesoftware.smack.Chat;
import org.jivesoftware.smack.XMPPException;
import org.jivesoftware.smack.packet.Message;

/**
 * @version $Revision: 563665 $
 */
public class XmppPrivateChatProducer extends DefaultProducer {
    private static final transient Log LOG = LogFactory.getLog(XmppPrivateChatProducer.class);
    private final XmppEndpoint endpoint;
    private final String participant;
    private Chat chat;

    public XmppPrivateChatProducer(XmppEndpoint endpoint, String participant) {
        super(endpoint);
        this.endpoint = endpoint;
        this.participant = participant;
        if (participant == null) {
            throw new IllegalArgumentException("No participant property specified");
        }
    }

    public void process(Exchange exchange) {
        // TODO it would be nice if we could reuse the message from the exchange
        Message message = chat.createMessage();
        message.setTo(participant);
        message.setFrom(endpoint.getUser());
        message.setThread(exchange.getExchangeId());
        message.setType(Message.Type.NORMAL);

        endpoint.getBinding().populateXmppMessage(message, exchange);
        if (LOG.isDebugEnabled()) {
            LOG.debug(">>>> message: " + message.getBody());
        }
        try {
            chat.sendMessage(message);
        } catch (XMPPException e) {
            throw new RuntimeXmppException(e);
        }
    }

    @Override
    protected void doStart() throws Exception {
        super.doStart();
        if (chat == null) {
            chat = endpoint.getConnection().createChat(getParticipant());
        }
    }

    @Override
    protected void doStop() throws Exception {
        chat = null;
        super.doStop();
    }

    // Properties
    // -------------------------------------------------------------------------
    public Chat getChat() {
        return chat;
    }

    public void setChat(Chat chat) {
        this.chat = chat;
    }

    public String getParticipant() {
        return participant;
    }
}
