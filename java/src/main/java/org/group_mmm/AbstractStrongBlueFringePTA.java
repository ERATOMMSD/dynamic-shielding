/*
 * Some part of this code is copied & pasted from RedBlueMerge.java in LearnLib 0.14.0, http://www.learnlib.de/.
 *
 * Since LearnLib is distributed under the Apache License, Version 2.0 (the "License"),
 * I understand that our usage, i.e., modification and redistribution is fine.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package org.group_mmm;

import de.learnlib.datastructure.pta.pta.AbstractBlueFringePTA;
import de.learnlib.datastructure.pta.pta.AbstractBlueFringePTAState;
import de.learnlib.datastructure.pta.pta.RedBlueMerge;

import javax.annotation.ParametersAreNonnullByDefault;

public class AbstractStrongBlueFringePTA<SP, TP, S extends AbstractBlueFringePTAState<SP, TP, S>> extends AbstractBlueFringePTA<SP, TP, S> {
    protected int min_depth;

    //@ requires 0 <= min_depth;
    public AbstractStrongBlueFringePTA(int alphabetSize, S root, int min_depth) {
        super(alphabetSize, root);
        assert min_depth >= 0 : "min_depth must be non-negative";
        this.min_depth = min_depth;
    }

    @Override
    @ParametersAreNonnullByDefault
    public RedBlueMerge<SP, TP, S> tryMerge(S qr, S qb) {
        RedBlueMerge<SP, TP, S> merge = new StrongRedBlueMerge<>(this, qr, qb, this.min_depth);
        if (!merge.merge()) {
            return null;
        }
        return merge;
    }
}
