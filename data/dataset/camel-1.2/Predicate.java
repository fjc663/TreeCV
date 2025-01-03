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
package org.apache.camel;

/**
 * Evaluates a binary <a
 * href="http://activemq.apache.org/camel/predicate.html">predicate</a> on the
 * message exchange to support things like <a
 * href="http://activemq.apache.org/camel/scripting-languages.html">scripting
 * languages</a>, <a href="http://activemq.apache.org/camel/xquery.html">XQuery</a>
 * or <a href="http://activemq.apache.org/camel/sql.html">SQL</a> as well as
 * any arbitrary Java expression.
 * 
 * @version $Revision: 563607 $
 */
public interface Predicate<E> {

    /**
     * Evaluates the predicate on the message exchange and returns true if this
     * exchange matches the predicate
     * 
     * @param exchange the message exchange
     * @return true if the predicate matches
     */
    boolean matches(E exchange);

    /**
     * Allows this predicate to be used nicely in testing to generate a nicely
     * formatted exception and message if this predicate does not match for the
     * given exchange.
     * 
     * @param text the description to use in the exception message
     * @param exchange the exchange to evaluate the expression on
     * @throws AssertionError if the predicate does not match
     */
    void assertMatches(String text, E exchange) throws AssertionError;

}
