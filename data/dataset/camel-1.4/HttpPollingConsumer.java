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
package org.apache.camel.component.http;

import java.io.IOException;
import java.io.InputStream;

import org.apache.camel.Message;
import org.apache.camel.RuntimeCamelException;
import org.apache.camel.component.http.helper.LoadingByteArrayOutputStream;
import org.apache.camel.impl.PollingConsumerSupport;
import org.apache.commons.httpclient.Header;
import org.apache.commons.httpclient.HttpClient;
import org.apache.commons.httpclient.HttpMethod;
import org.apache.commons.httpclient.methods.GetMethod;
import org.apache.commons.io.IOUtils;

/**
 * A polling HTTP consumer which by default performs a GET
 *
 * @version $Revision: 651239 $
 */
public class HttpPollingConsumer extends PollingConsumerSupport<HttpExchange> {
    private final HttpEndpoint endpoint;
    private HttpClient httpClient;

    public HttpPollingConsumer(HttpEndpoint endpoint) {
        super(endpoint);
        this.endpoint = endpoint;
        httpClient = endpoint.createHttpClient();
    }

    public HttpExchange receive() {
        return receiveNoWait();
    }

    public HttpExchange receive(long timeout) {
        return receiveNoWait();
    }

    public HttpExchange receiveNoWait() {
        HttpExchange exchange = endpoint.createExchange();
        HttpMethod method = createMethod();

        try {
            int responseCode = httpClient.executeMethod(method);
            // lets store the result in the output message.
            LoadingByteArrayOutputStream bos = new LoadingByteArrayOutputStream();
            InputStream is = method.getResponseBodyAsStream();
            IOUtils.copy(is, bos);
            bos.flush();
            is.close();
            Message message = exchange.getIn();
            message.setBody(bos.createInputStream());

            // lets set the headers
            Header[] headers = method.getResponseHeaders();
            for (Header header : headers) {
                String name = header.getName();
                String value = header.getValue();
                message.setHeader(name, value);
            }

            message.setHeader("http.responseCode", responseCode);
            return exchange;
        } catch (IOException e) {
            throw new RuntimeCamelException(e);
        } finally {
            method.releaseConnection();
        }
    }

    // Properties
    //-------------------------------------------------------------------------
    public HttpClient getHttpClient() {
        return httpClient;
    }

    public void setHttpClient(HttpClient httpClient) {
        this.httpClient = httpClient;
    }

    // Implementation methods
    //-------------------------------------------------------------------------
    protected HttpMethod createMethod() {
        String uri = endpoint.getEndpointUri();
        return new GetMethod(uri);
    }

    protected void doStart() throws Exception {
    }

    protected void doStop() throws Exception {
    }
}
