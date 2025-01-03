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
package org.apache.camel.processor.interceptor;

import org.apache.camel.Processor;
import org.apache.camel.model.ProcessorType;
import org.apache.camel.spi.InterceptStrategy;

/**
 * An interceptor strategy for tracing routes
 *
 * @version $Revision: 675876 $
 */
public class Tracer implements InterceptStrategy {

    private TraceFormatter formatter = new TraceFormatter();

    public Processor wrapProcessorInInterceptors(ProcessorType processorType, Processor target) throws Exception {
        // Force the creation of an id, otherwise the id is not available when the trace formatter is
        // outputting trace information
        String id = processorType.idOrCreate();
        return new TraceInterceptor(processorType, target, formatter);
    }

    public TraceFormatter getFormatter() {
        return formatter;
    }

    public void setFormatter(TraceFormatter formatter) {
        this.formatter = formatter;
    }
}
