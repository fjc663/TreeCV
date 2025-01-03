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
package org.apache.camel.language.bean;

import org.apache.camel.Exchange;
import org.apache.camel.Expression;
import org.apache.camel.Predicate;
import org.apache.camel.builder.PredicateBuilder;
import org.apache.camel.spi.Language;
import org.apache.camel.util.ObjectHelper;

/**
 * A <a href="http://activemq.apache.org/camel/bean-language.html>bean language</a>
 * which uses a simple text notation to invoke methods on beans to evaluate predicates or expressions<p/>
 * <p/>
 * The notation is essentially <code>beanName.methodName</code> which is then invoked using the
 * beanName to lookup in the <a href="http://activemq.apache.org/camel/registry.html>registry</a>
 * then the method is invoked to evaluate the expression using the
 * <a href="http://activemq.apache.org/camel/bean-integration.html>bean integration</a> to bind the
 * {@link Exchange} to the method arguments.
 *
 * @version $Revision: 630591 $
 */
public class BeanLanguage implements Language {
    public Predicate<Exchange> createPredicate(String expression) {
        return PredicateBuilder.toPredicate(createExpression(expression));
    }

    public Expression<Exchange> createExpression(String expression) {
        ObjectHelper.notNull(expression, "expression");

        int idx = expression.lastIndexOf('.');
        String beanName = expression;
        String method = null;
        if (idx > 0) {
            beanName = expression.substring(0, idx);
            method = expression.substring(idx + 1);
        }
        return new BeanExpression(beanName, method);
    }
}
