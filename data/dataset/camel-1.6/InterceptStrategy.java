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
package org.apache.camel.spi;

import org.apache.camel.Processor;
import org.apache.camel.model.ProcessorType;

 /**
  * The purpose of this interface is to allow an implementation to wrap
  * processors in a route with interceptors.  For example, a possible
  * usecase is to gather performance statistics at the processor's level.
  *
  * @version $Revision: 679971 $
  */
public interface InterceptStrategy {

    /**
     * This method is invoked by
     * {@link ProcessorType#wrapProcessor(RouteContext, Processor)}
     * to give the implementor an opportunity to wrap the target processor
     * in a route.
     *
     * @param processorType the object that invokes this method
     * @param target the processor to be wrapped
     * @return processor wrapped with an interceptor or not wrapped
     * @throws Exception can be thrown
     */
    Processor wrapProcessorInInterceptors(ProcessorType processorType,
            Processor target) throws Exception;
}
