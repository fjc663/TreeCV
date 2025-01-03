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
package org.apache.camel.builder;

import java.util.ArrayList;
import java.util.List;

import org.apache.camel.model.ExceptionType;
import org.apache.camel.processor.ErrorHandlerSupport;

/**
 * Base class for builders of error handling.
 *
 * @version $Revision: 673837 $
 */
public abstract class ErrorHandlerBuilderSupport implements ErrorHandlerBuilder {
    private List<ExceptionType> exceptions = new ArrayList<ExceptionType>();

    public void addErrorHandlers(ExceptionType exception) {
        exceptions.add(exception);
    }

    protected void configure(ErrorHandlerSupport handler) {
        for (ExceptionType exception : exceptions) {
            handler.addExceptionPolicy(exception);
        }
    }

    public List<ExceptionType> getExceptions() {
        return exceptions;
    }
}
