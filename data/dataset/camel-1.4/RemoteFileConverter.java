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
package org.apache.camel.component.file.remote;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;

import org.apache.camel.Converter;

/**
 * A set of converter methods for working with remote file types
 *
 * @version $Revision: 647893 $
 */
@Converter
public final class RemoteFileConverter {
    private RemoteFileConverter() {
        // Helper Class
    }

    @Converter
    public static byte[] toByteArray(ByteArrayOutputStream os) {
        return os.toByteArray();
    }

    @Converter
    public static String toString(ByteArrayOutputStream os) {
        return os.toString();
    }

    @Converter
    public static InputStream toInputStream(ByteArrayOutputStream os) {
        return new ByteArrayInputStream(os.toByteArray());
    }

}
