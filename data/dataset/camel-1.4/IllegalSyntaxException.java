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
package org.apache.camel.language;

import org.apache.camel.RuntimeCamelException;
import org.apache.camel.spi.Language;

/**
 * An exception thrown if some illegal syntax is rejected by a specific language
 *
 * @version $Revision: 630568 $
 */
public class IllegalSyntaxException extends RuntimeCamelException {
    private final Language language;
    private final String expression;

    public IllegalSyntaxException(Language language, String expression) {
        super("Illegal syntax for language: " + language + ". Expression: " + expression);
        this.language = language;
        this.expression = expression;
    }

    public String getExpression() {
        return expression;
    }

    public Language getLanguage() {
        return language;
    }
}
