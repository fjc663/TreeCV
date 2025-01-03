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
package org.apache.camel.component.mina;

import java.net.SocketAddress;

import org.apache.camel.CamelException;
import org.apache.camel.Processor;
import org.apache.camel.impl.DefaultConsumer;
import org.apache.camel.util.ExchangeHelper;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.mina.common.IoAcceptor;
import org.apache.mina.common.IoHandler;
import org.apache.mina.common.IoHandlerAdapter;
import org.apache.mina.common.IoSession;

/**
 * A {@link org.apache.camel.Consumer Consumer} implementation for Apache MINA.
 *
 * @version $Revision: 641152 $
 */
public class MinaConsumer extends DefaultConsumer<MinaExchange> {
    private static final transient Log LOG = LogFactory.getLog(MinaConsumer.class);

    private final MinaEndpoint endpoint;
    private final SocketAddress address;
    private final IoAcceptor acceptor;

    public MinaConsumer(final MinaEndpoint endpoint, Processor processor) {
        super(endpoint, processor);
        this.endpoint = endpoint;
        this.address = endpoint.getAddress();
        this.acceptor = endpoint.getAcceptor();
    }

    @Override
    protected void doStart() throws Exception {
        super.doStart();
        if (LOG.isDebugEnabled()) {
            LOG.debug("Binding to server address: " + address + " using acceptor: " + acceptor);
        }

        IoHandler handler = new ReceiveHandler();
        acceptor.bind(address, handler, endpoint.getAcceptorConfig());
    }

    @Override
    protected void doStop() throws Exception {
        if (LOG.isDebugEnabled()) {
            LOG.debug("Unbinding from server address: " + address + " using acceptor: " + acceptor);
        }
        acceptor.unbind(address);
        super.doStop();
    }

    /**
     * Handles comsuming messages and replying if the exchange is out capable.
     */
    private final class ReceiveHandler extends IoHandlerAdapter {

        @Override
        public void exceptionCaught(IoSession session, Throwable cause) throws Exception {
            // close invalid session
            if (session != null) {
                LOG.debug("Closing session as an exception was thrown from MINA");
                session.close();
            }

            // must wrap and rethrow since cause can be of Throwable and we must only throw Exception
            throw new CamelException(cause);
        }

        @Override
        public void messageReceived(IoSession session, Object object) throws Exception {
            if (LOG.isDebugEnabled()) {
                LOG.debug("Received body: " + object);
            }

            MinaExchange exchange = endpoint.createExchange(session, object);
            getProcessor().process(exchange);

            if (ExchangeHelper.isOutCapable(exchange)) {
                Object body = MinaPayloadHelper.getOut(endpoint, exchange);
                boolean failed = exchange.isFailed();

                if (failed) {
                    // can not write a response since the exchange is failed and we don't know in what state the
                    // in/out messages are in so the session is closed
                    LOG.warn("Can not write body since the exchange is failed, closing session: " + exchange);
                    session.close();
                } else if (body == null) {
                    // must close session if no data to write otherwise client will never receive a response
                    // and wait forever (if not timing out)
                    LOG.warn("Can not write body since its null, closing session: " + exchange);
                    session.close();
                } else {
                    // we got a response to write
                    if (LOG.isDebugEnabled()) {
                        LOG.debug("Writing body: " + body);
                    }
                    MinaHelper.writeBody(session, body, exchange);
                }
            } else {
                if (LOG.isDebugEnabled()) {
                    LOG.debug("Can not write body since this exchange is not out capable: " + exchange);
                }
            }
        }

    }

}
