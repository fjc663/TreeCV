/*
 * The Apache Software License, Version 1.1
 *
 *
 * Copyright (c) 1999,2000 The Apache Software Foundation.  All rights 
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
 *    if any, must include the following acknowledgment:  
 *       "This product includes software developed by the
 *        Apache Software Foundation (http://www.apache.org/)."
 *    Alternately, this acknowledgment may appear in the software itself,
 *    if and wherever such third-party acknowledgments normally appear.
 *
 * 4. The names "Xerces" and "Apache Software Foundation" must
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
 * originally based on software copyright (c) 1999, International
 * Business Machines, Inc., http://www.apache.org.  For more
 * information on the Apache Software Foundation, please see
 * <http://www.apache.org/>.
 */
package org.apache.wml.dom;

import org.apache.wml.*;

/**
 * @version $Id$
 * @author <a href="mailto:david@topware.com.tw">David Li</a>
 */

public class WMLInputElementImpl extends WMLElementImpl implements WMLInputElement {

    public WMLInputElementImpl (WMLDocumentImpl owner, String tagName) {
	super( owner, tagName);
    }

    public void setSize(int newValue) {
	setAttribute("size", newValue);
    }

    public int getSize() {
	return getAttribute("size", 0);
    }

    public void setFormat(String newValue) {
	setAttribute("format", newValue);
    }

    public String getFormat() {
	return getAttribute("format");
    }

    public void setValue(String newValue) {
	setAttribute("value", newValue);
    }

    public String getValue() {
	return getAttribute("value");
    }

    public void setMaxLength(int newValue) {
	setAttribute("maxlength", newValue);
    }

    public int getMaxLength() {
	return getAttribute("maxlength", 0);
    }

    public void setTabIndex(int newValue) {
	setAttribute("tabindex", newValue);
    }

    public int getTabIndex() {
	return getAttribute("tabindex", 0);
    }

    public void setClassName(String newValue) {
	setAttribute("class", newValue);
    }

    public String getClassName() {
	return getAttribute("class");
    }

    public void setXmlLang(String newValue) {
	setAttribute("xml:lang", newValue);
    }

    public String getXmlLang() {
	return getAttribute("xml:lang");
    }

    public void setEmptyOk(boolean newValue) {
	setAttribute("emptyok", newValue);
    }

    public boolean getEmptyOk() {
	return getAttribute("emptyok", false);
    }

    public void setTitle(String newValue) {
	setAttribute("title", newValue);
    }

    public String getTitle() {
	return getAttribute("title");
    }

    public void setId(String newValue) {
	setAttribute("id", newValue);
    }

    public String getId() {
	return getAttribute("id");
    }

    public void setType(String newValue) {
	setAttribute("type", newValue);
    }

    public String getType() {
	return getAttribute("type");
    }

    public void setName(String newValue) {
	setAttribute("name", newValue);
    }

    public String getName() {
	return getAttribute("name");
    }

}