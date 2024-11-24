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
package org.apache.camel.processor.resequencer;

/**
 * A strategy for comparing {@link Element} instances. This strategy uses
 * another {@link SequenceElementComparator} instance for comparing elements
 * contained by {@link Element} instances.
 * 
 * @author Martin Krasser
 * 
 * @version $Revision
 */
class ElementComparator<E> implements SequenceElementComparator<Element<E>> {

    /**
     * A sequence element comparator this comparator delegates to.
     */
    private SequenceElementComparator<E> comparator;
    
    /**
     * Creates a new element comparator instance.
     * 
     * @param comparator a sequence element comparator this comparator delegates
     *        to.
     */
    public ElementComparator(SequenceElementComparator<E> comparator) {
        this.comparator = comparator;
    }
    
    /**
     * @see SequenceElementComparator#predecessor(java.lang.Object, java.lang.Object)
     */
    public boolean predecessor(Element<E> o1, Element<E> o2) {
        return comparator.predecessor(o1.getObject(), o2.getObject());
    }

    /**
     * @see SequenceElementComparator#successor(java.lang.Object, java.lang.Object)
     */
    public boolean successor(Element<E> o1, Element<E> o2) {
        return comparator.successor(o1.getObject(), o2.getObject());
    }

    /**
     * @see java.util.Comparator#compare(java.lang.Object, java.lang.Object)
     */
    public int compare(Element<E> o1, Element<E> o2) {
        return comparator.compare(o1.getObject(), o2.getObject());
    }

}
