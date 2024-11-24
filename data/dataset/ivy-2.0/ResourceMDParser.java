/*
 *  Licensed to the Apache Software Foundation (ASF) under one or more
 *  contributor license agreements.  See the NOTICE file distributed with
 *  this work for additional information regarding copyright ownership.
 *  The ASF licenses this file to You under the Apache License, Version 2.0
 *  (the "License"); you may not use this file except in compliance with
 *  the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */
package org.apache.ivy.plugins.resolver.util;

import org.apache.ivy.plugins.repository.Resource;

public interface ResourceMDParser {
    /**
     * Parses the module descriptor designed by the given resource.
     * 
     * @param resource
     *            the resource at which the module descriptor is located
     * @param rev
     *            the revision at which the module descriptor should be
     * @return the parsed module descriptor as a {@link MDResolvedResource}, or <code>null</code>
     *         if parsing has failed or if the resource is not available.
     */
    MDResolvedResource parse(Resource resource, String rev);
}
