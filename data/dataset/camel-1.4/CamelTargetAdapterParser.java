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
package org.apache.camel.component.spring.integration.adapter.config;

import org.w3c.dom.Attr;
import org.w3c.dom.Element;
import org.w3c.dom.NamedNodeMap;

import org.apache.camel.component.spring.integration.adapter.CamelTargetAdapter;
import org.springframework.beans.factory.parsing.BeanComponentDefinition;
import org.springframework.beans.factory.support.BeanDefinitionBuilder;
import org.springframework.beans.factory.support.RootBeanDefinition;
import org.springframework.beans.factory.xml.AbstractSimpleBeanDefinitionParser;
import org.springframework.beans.factory.xml.ParserContext;
import org.springframework.integration.ConfigurationException;
import org.springframework.integration.endpoint.HandlerEndpoint;
import org.springframework.util.StringUtils;

/**
 * Parser for the &lt;camelTarget/&gt; element
 * @author Willem Jiang
 *
 * @version $Revision: 677135 $
 */

public class CamelTargetAdapterParser extends AbstractCamelContextBeanDefinitionParaser {
    protected Class<?> getBeanClass(Element element) {
        return HandlerEndpoint.class;
    }

    protected boolean shouldGenerateId() {
        return false;
    }

    protected boolean shouldGenerateIdAsFallback() {
        return true;
    }

    protected void parseAttributes(Element element, ParserContext ctx, BeanDefinitionBuilder bean) {
        NamedNodeMap atts = element.getAttributes();
        for (int i = 0; i < atts.getLength(); i++) {
            Attr node = (Attr) atts.item(i);
            String val = node.getValue();
            String name = node.getLocalName();
            if (!name.equals("requestChannel") && !name.equals("replyChannel")) {
                mapToProperty(bean, name, val);
            }
        }
    }

    protected void doParse(Element element, ParserContext parserContext, BeanDefinitionBuilder builder) {
        BeanDefinitionBuilder adapterDefBuilder = BeanDefinitionBuilder.rootBeanDefinition(CamelTargetAdapter.class);
        String requestChannel = element.getAttribute("requestChannel");
        String replyChannel = element.getAttribute("replyChannel");
        // Check the requestChannel
        if (!StringUtils.hasText(requestChannel)) {
            throw new ConfigurationException("The 'requestChannel' attribute is required.");
        }
        // Set the adapter bean's property
        parseAttributes(element, parserContext, adapterDefBuilder);
        parseCamelContext(element, parserContext, adapterDefBuilder);

        String adapterBeanName = parserContext.getReaderContext().generateBeanName(adapterDefBuilder.getBeanDefinition());
        parserContext.registerBeanComponent(new BeanComponentDefinition(adapterDefBuilder.getBeanDefinition(), adapterBeanName));
        builder.addConstructorArgReference(adapterBeanName);
        builder.addPropertyValue("inputChannelName", requestChannel);
        if (StringUtils.hasText(replyChannel)) {
            builder.addPropertyValue("outputChannelName", replyChannel);
        }
    }

}
