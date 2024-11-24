/*
 * The Apache Software License, Version 1.1
 *
 *
 * Copyright (c) 2002-2003 The Apache Software Foundation.  All rights 
 * reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer. 
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * 3. The end-user documentation included with the redistribution,
 *    if any must include the following acknowledgment:  
 *       "This product includes software developed by the
 *        Apache Software Foundation (http://www.apache.org/)."
 *    Alternately, this acknowledgment may appear in the software itself,
 *    if and wherever such third-party acknowledgments normally appear.
 *
 * 4. The names "Xalan" and "Apache Software Foundation" must
 *    not be used to endorse or promote products derived from this
 *    software without prior written permission. For written 
 *    permission, please contact apache@apache.org.
 *
 * 5. Products derived from this software may not be called "Apache",
 *    nor may "Apache" appear in their name, without prior written
 *    permission of the Apache Software Foundation.
 *
 * THIS SOFTWARE IS PROVIDED ``AS IS'' AND ANY EXPRESSED OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED.  IN NO EVENT SHALL THE APACHE SOFTWARE FOUNDATION OR
 * ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 * USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * ====================================================================
 *
 * This software consists of voluntary contributions made by many
 * individuals on behalf of the Apache Software Foundation and was
 * originally based on software copyright (c) 1999, Lotus
 * Development Corporation., http://www.lotus.com.  For more
 * information on the Apache Software Foundation, please see
 * <http://www.apache.org/>.
 */


package org.apache.xpath.domapi;

import javax.xml.transform.TransformerException;

import org.apache.xalan.res.XSLMessages;
import org.apache.xml.utils.PrefixResolver;
import org.apache.xpath.XPath;
import org.apache.xpath.XPathContext;
import org.apache.xpath.objects.XObject;
import org.apache.xpath.res.XPATHErrorResources;

import org.w3c.dom.DOMException;
import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.w3c.dom.xpath.XPathException;
import org.w3c.dom.xpath.XPathExpression;
import org.w3c.dom.xpath.XPathNamespace;

/**
 * 
 * The class provides an implementation of XPathExpression according 
 * to the DOM L3 XPath Specification, Working Draft 28, March 2002.
 *
 * <p>See also the <a href='http://www.w3.org/TR/2002/WD-DOM-Level-3-XPath-20020328'>Document Object Model (DOM) Level 3 XPath Specification</a>.</p>

 * <p>The <code>XPathExpression</code> interface represents a parsed and resolved 
 * XPath expression.</p>
 * 
 * @see org.w3c.dom.xpath.XPathExpression
 */
public class XPathExpressionImpl implements XPathExpression {

  private PrefixResolver m_resolver;      
  
  /**
   * The xpath object that this expression wraps
   */
  private XPath m_xpath;
  
  /**
   * The document to be searched to parallel the case where the XPathEvaluator
   * is obtained by casting a Document.
   */  
  private Document m_doc = null;  

    /**
     * Constructor for XPathExpressionImpl.
     * 
     * @param xpath The wrapped XPath object.
     * @param doc The document to be searched, to parallel the case where''
     *            the XPathEvaluator is obtained by casting the document.
     */
    XPathExpressionImpl(XPath xpath, Document doc) {
        m_xpath = xpath;
        m_doc = doc;
    }

    /**
     * <meta name="usage" content="experimental"/>
     *
     * This method provides an implementation XPathResult.evaluate according 
     * to the DOM L3 XPath Specification, Working Draft 28, March 2002.
     *
     * <p>See also the <a href='http://www.w3.org/TR/2002/WD-DOM-Level-3-XPath-20020328'>Document Object Model (DOM) Level 3 XPath Specification</a>.</p>
     * 
     * <p>Evaluates this XPath expression and returns a result.</p>
     * @param contextNode The <code>context</code> is context node for the 
     *   evaluation of this XPath expression.If the XPathEvaluator was 
     *   obtained by casting the <code>Document</code> then this must be 
     *   owned by the same document and must be a <code>Document</code>, 
     *   <code>Element</code>, <code>Attribute</code>, <code>Text</code>, 
     *   <code>CDATASection</code>, <code>Comment</code>, 
     *   <code>ProcessingInstruction</code>, or <code>XPathNamespace</code> 
     *   node.If the context node is a <code>Text</code> or a 
     *   <code>CDATASection</code>, then the context is interpreted as the 
     *   whole logical text node as seen by XPath, unless the node is empty 
     *   in which case it may not serve as the XPath context.
     * @param type If a specific <code>type</code> is specified, then the 
     *   result will be coerced to return the specified type relying on 
     *   XPath conversions and fail if the desired coercion is not possible. 
     *   This must be one of the type codes of <code>XPathResult</code>.
    *  @param result The <code>result</code> specifies a specific result 
     *   object which may be reused and returned by this method. If this is 
     *   specified as <code>null</code>or the implementation does not reuse 
     *   the specified result, a new result object will be constructed and 
     *   returned.For XPath 1.0 results, this object will be of type 
     *   <code>XPathResult</code>.
     * @return The result of the evaluation of the XPath expression.For XPath 
     *   1.0 results, this object will be of type <code>XPathResult</code>.
     * @exception XPathException
     *   TYPE_ERR: Raised if the result cannot be converted to return the 
     *   specified type.
     * @exception DOMException
     *   WRONG_DOCUMENT_ERR: The Node is from a document that is not supported 
     *   by the XPathEvaluator that created this 
     *   <code>XPathExpression</code>.
     *   <br>NOT_SUPPORTED_ERR: The Node is not a type permitted as an XPath 
     *   context node.   
     * 
     * @see org.w3c.dom.xpath.XPathExpression#evaluate(Node, short, XPathResult)
     */
    public Object evaluate(
        Node contextNode,
        short type,
        Object result)
        throws XPathException, DOMException {
            
        // If the XPathEvaluator was determined by "casting" the document    
        if (m_doc != null) {
        
            // Check that the context node is owned by the same document
            if ((contextNode != m_doc) && (!contextNode.getOwnerDocument().equals(m_doc))) {
                String fmsg = XSLMessages.createXPATHMessage(XPATHErrorResources.ER_WRONG_DOCUMENT, null);       
                throw new DOMException(DOMException.WRONG_DOCUMENT_ERR, fmsg);
            }
            
            // Check that the context node is an acceptable node type
            short nodeType = contextNode.getNodeType();
            if ((nodeType != Document.DOCUMENT_NODE) &&
                (nodeType != Document.ELEMENT_NODE) && 
                (nodeType != Document.ATTRIBUTE_NODE) &&
                (nodeType != Document.TEXT_NODE) &&
                (nodeType != Document.CDATA_SECTION_NODE) &&
                (nodeType != Document.COMMENT_NODE) &&
                (nodeType != Document.PROCESSING_INSTRUCTION_NODE) &&
                (nodeType != XPathNamespace.XPATH_NAMESPACE_NODE)) {
                    String fmsg = XSLMessages.createXPATHMessage(XPATHErrorResources.ER_WRONG_NODETYPE, null);       
                    throw new DOMException(DOMException.NOT_SUPPORTED_ERR, fmsg);
            }
        }
            
        //     
        // If the type is not a supported type, throw an exception and be
        // done with it!
        if (!XPathResultImpl.isValidType(type)) {
            String fmsg = XSLMessages.createXPATHMessage(XPATHErrorResources.ER_INVALID_XPATH_TYPE, new Object[] {new Integer(type)});       
            throw new XPathException(XPathException.TYPE_ERR,fmsg); // Invalid XPath type argument: {0}               
        }
        
        // Cache xpath context?
        XPathContext xpathSupport = new XPathContext();
        
        // if m_document is not null, build the DTM from the document 
        if (null != m_doc) {
            xpathSupport.getDTMHandleFromNode(m_doc);
        }

        XObject xobj = null;
        try {
            xobj = m_xpath.execute(xpathSupport, contextNode, m_resolver );         
        } catch (TransformerException te) {
            // What should we do here?
            throw new XPathException(XPathException.INVALID_EXPRESSION_ERR,te.getMessageAndLocation()); 
        }

        // Create a new XPathResult object
        // Reuse result object passed in?
        // The constructor will check the compatibility of type and xobj and
        // throw an exception if they are not compatible.
        return new XPathResultImpl(type,xobj,contextNode);
    }

}
