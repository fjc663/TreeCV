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
package org.apache.camel.model;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlElements;
import javax.xml.bind.annotation.XmlRootElement;

import org.apache.camel.Processor;
import org.apache.camel.model.dataformat.ArtixDSDataFormat;
import org.apache.camel.model.dataformat.CsvDataFormat;
import org.apache.camel.model.dataformat.DataFormatType;
import org.apache.camel.model.dataformat.JaxbDataFormat;
import org.apache.camel.model.dataformat.SerializationDataFormat;
import org.apache.camel.model.dataformat.StringDataFormat;
import org.apache.camel.model.dataformat.XMLBeansDataFormat;
import org.apache.camel.processor.MarshalProcessor;
import org.apache.camel.spi.DataFormat;
import org.apache.camel.spi.RouteContext;

/**
 * Marshals to a binary payload using the given {@link DataFormatType}
 *
 * @version $Revision: 671918 $
 */
@XmlRootElement(name = "marshal")
@XmlAccessorType(XmlAccessType.FIELD)
public class MarshalType extends OutputType<ProcessorType> {
    @XmlAttribute(required = false)
    private String ref;
    // TODO cannot use @XmlElementRef as it doesn't allow optional properties
    // @XmlElementRef
    @XmlElements({
    @XmlElement(required = false, name = "artixDS", type = ArtixDSDataFormat.class),
    @XmlElement(required = false, name = "csv", type = CsvDataFormat.class),
    @XmlElement(required = false, name = "jaxb", type = JaxbDataFormat.class),
    @XmlElement(required = false, name = "serialization", type = SerializationDataFormat.class),
    @XmlElement(required = false, name = "string", type = StringDataFormat.class),
    @XmlElement(required = false, name = "xmlBeans", type = XMLBeansDataFormat.class)}
    )
    private DataFormatType dataFormatType;

    public MarshalType() {
    }

    public MarshalType(DataFormatType dataFormatType) {
        this.dataFormatType = dataFormatType;
    }

    public MarshalType(String ref) {
        this.ref = ref;
    }

    @Override
    public String toString() {
        if (dataFormatType != null) {
            return "Marshal[" + dataFormatType + "]";
        } else {
            return "Marshal[ref:  " + ref + "]";
        }
    }

    @Override
    public String getShortName() {
        return "marshal";
    }

    public String getRef() {
        return ref;
    }

    public void setRef(String ref) {
        this.ref = ref;
    }

    public DataFormatType getDataFormatType() {
        return dataFormatType;
    }

    public void setDataFormatType(DataFormatType dataFormatType) {
        this.dataFormatType = dataFormatType;
    }

    @Override
    public Processor createProcessor(RouteContext routeContext) {
        DataFormat dataFormat = DataFormatType.getDataFormat(routeContext, getDataFormatType(), ref);
        return new MarshalProcessor(dataFormat);
    }
}
