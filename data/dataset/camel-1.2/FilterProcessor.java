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
package org.apache.camel.processor;

import org.apache.camel.Exchange;
import org.apache.camel.Predicate;
import org.apache.camel.Processor;

/**
 * @version $Revision: 563607 $
 */
public class FilterProcessor extends DelegateProcessor {
    private Predicate<Exchange> predicate;

    public FilterProcessor(Predicate<Exchange> predicate, Processor processor) {
        super(processor);
        this.predicate = predicate;
    }

    public void process(Exchange exchange) throws Exception {
        if (predicate.matches(exchange)) {
            super.process(exchange);
        }
    }

    @Override
    public String toString() {
        return "Filter[if: " + predicate + " do: " + getProcessor() + "]";
    }

    public Predicate<Exchange> getPredicate() {
        return predicate;
    }
}
