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
package org.apache.camel.component.cxf;

/**
 * Constants used in this module
 *
 * @version $Revision: 583092 $
 */
public interface CxfConstants {
    String METHOD = "method";
    String SEI = "sei";
    String IMPL = "impl";
    String WSDL_URL = "wsdlURL";
    String ADDRESS = "address";
    String SERVICE_NAME = "serviceName";
    String PORT_NAME = "portName";
    String PROTOCOL_NAME_RES = "res";
    String OPERATION_NAME = "operationName";
    String SPRING_CONTEXT_ENDPOINT = "bean:";
    // service name -- come from the wsdl   
}


