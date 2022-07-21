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
import de.learnlib.datastructure.pta.pta.BlueFringePTA;
import de.learnlib.datastructure.pta.pta.BlueFringePTAState;
import de.learnlib.datastructure.pta.pta.RedBlueMerge;

import javax.annotation.ParametersAreNonnullByDefault;

public class StrongBlueFringePTA<SP, TP> extends AbstractBlueFringePTA<SP, TP, StrongBlueFringePTAState<SP, TP>> {
    protected int min_depth;

    //@ requires 0 <= min_depth;
    public StrongBlueFringePTA(int alphabetSize, int min_depth) {
        super(alphabetSize, new StrongBlueFringePTAState<>());
        assert min_depth >= 0 : "min_depth must be non-negative";
        this.min_depth = min_depth;
    }

    @Override
    @ParametersAreNonnullByDefault
    public RedBlueMerge<SP, TP, StrongBlueFringePTAState<SP, TP>> tryMerge(StrongBlueFringePTAState<SP, TP> qr, StrongBlueFringePTAState<SP, TP> qb) {
        RedBlueMerge<SP, TP, StrongBlueFringePTAState<SP, TP>> merge = new StrongRedBlueMerge<>(this, qr, qb, this.min_depth);
        if (!merge.merge()) {
            return null;
        }
        return merge;
    }
}
