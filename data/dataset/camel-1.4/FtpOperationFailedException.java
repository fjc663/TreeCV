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

import org.apache.camel.RuntimeCamelException;

/**
 * @version $Revision: 630574 $
 */
public class FtpOperationFailedException extends RuntimeCamelException {
    private final int code;
    private final String reason;

    public FtpOperationFailedException(int code, String reason) {
        super("Ftp Operation failed: " + reason.trim() + " Code: " + code);
        this.code = code;
        this.reason = reason;
    }

    /**
     * Return the failure code
     */
    public int getCode() {
        return code;
    }

    /**
     * Return the failure reason
     */
    public String getReason() {
        return reason;
    }
}
