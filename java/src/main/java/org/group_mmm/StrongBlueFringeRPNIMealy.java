/*
 * Some part of this code is copied & pasted from BlueFringeRPNIMealy.java in LearnLib 0.14.0, http://www.learnlib.de/.
 *
 * Since LearnLib is distributed under the Apache License, Version 2.0 (the "License"),
 * I understand that our usage, i.e., modification and redistribution is fine.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

package org.group_mmm;

import de.learnlib.algorithms.rpni.BlueFringeRPNIMealy;
import de.learnlib.api.query.DefaultQuery;
import de.learnlib.datastructure.pta.pta.AbstractBlueFringePTA;
import de.learnlib.datastructure.pta.pta.BlueFringePTA;
import de.learnlib.datastructure.pta.pta.PTATransition;
import de.learnlib.datastructure.pta.pta.RedBlueMerge;
import lombok.Setter;
import net.automatalib.automata.transducers.MealyMachine;
import net.automatalib.automata.transducers.impl.compact.CompactMealy;
import net.automatalib.commons.util.Pair;
import net.automatalib.words.Alphabet;
import net.automatalib.words.Word;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class StrongBlueFringeRPNIMealy<I, O> extends BlueFringeRPNIMealy<I, O> {
    @Setter
    protected int min_depth;
    protected int skipMealySize;
    private final List<Pair<int[], Word<O>>> samples = new ArrayList<>();

    /**
     * @param alphabet      The input alphabet of the Mealy machine
     * @param minDepth      Threshold of the merging
     * @param skipMealySize We do not merge the states if the Mealy machine is smaller than this.
     */
    //@ requires 0 <= minDepth;
    public StrongBlueFringeRPNIMealy(Alphabet<I> alphabet, int minDepth, int skipMealySize) {
        super(alphabet);
        this.min_depth = minDepth;
        this.skipMealySize = skipMealySize;
    }

    @Override
    public void addSamples(Collection<? extends DefaultQuery<I, Word<O>>> samples) {
        for (DefaultQuery<I, Word<O>> sample : samples) {
            this.samples.add(Pair.of(sample.getInput().toIntArray(this.alphabet), sample.getOutput()));
        }
    }

    public List<Pair<List<I>, List<O>>> getSamples() {
        return this.samples.stream().map(pair -> Pair.of(
                Arrays.stream(pair.getFirst()).mapToObj(this.alphabet::getSymbol).collect(Collectors.toList()),
                pair.getSecond().asList())).collect(Collectors.toList());
    }

    protected void initializePTAGeneric(AbstractBlueFringePTA<Void, O, ?> pta) {
        for (Pair<int[], Word<O>> sample : this.samples) {
            pta.addSampleWithTransitionProperties(sample.getFirst(), sample.getSecond().asList());
        }
    }

    @Override
    protected void initializePTA(BlueFringePTA<Void, O> pta) {
        for (Pair<int[], Word<O>> sample : this.samples) {
            pta.addSampleWithTransitionProperties(sample.getFirst(), sample.getSecond().asList());
        }
    }

    protected MealyMachine<?, I, ?, O> computeModelWithMinDepth() {
        StrongBlueFringePTA<Void, O> pta = new StrongBlueFringePTA<>(alphabetSize, min_depth);
        initializePTAGeneric(pta);

        Queue<PTATransition<StrongBlueFringePTAState<Void, O>>> blue = order.createWorklist();

        pta.init(blue::offer);

        PTATransition<StrongBlueFringePTAState<Void, O>> qbRef;
        while ((qbRef = blue.poll()) != null) {
            StrongBlueFringePTAState<Void, O> qb = qbRef.getTarget();
            assert qb != null;
            // We stop state merging once the Mealy machine becomes small enough.
            if (pta.size() < this.skipMealySize) {
                pta.promote(qb, blue::offer);
                continue;
            }

            Stream<StrongBlueFringePTAState<Void, O>> stream = pta.redStatesStream();
            if (parallel) {
                stream = stream.parallel();
            }
            // This is a hack because of the strong state merge
            qb.forEachSucc((s) -> s.setParent(qb));
            if (qb.getSuccessors() == null || qb.getSuccessors().size() < 2 || StrongRedBlueMerge.notEnouchChildrenDepth(qb, min_depth, alphabetSize)) {
                pta.promote(qb, blue::offer);
                continue;
            }
            @SuppressWarnings("nullness") // we filter the null merges
            Stream<RedBlueMerge<Void, O, StrongBlueFringePTAState<Void, O>>> filtered =
                    stream.map(qr -> tryMerge(pta, qr, qb)).filter(Objects::nonNull);

            Optional<RedBlueMerge<Void, O, StrongBlueFringePTAState<Void, O>>> result =
                    (deterministic) ? filtered.findFirst() : filtered.findAny();

            if (result.isPresent()) {
                RedBlueMerge<Void, O, StrongBlueFringePTAState<Void, O>> mod = result.get();
                mod.apply(pta, blue::offer);
            } else {
                pta.promote(qb, blue::offer);
            }
        }

        return ptaToModel(pta);
    }

    @Override
    public MealyMachine<?, I, ?, O> computeModel() {
        if (this.min_depth > 0) {
            return this.computeModelWithMinDepth();
        } else {
            return super.computeModel();
        }
    }

    protected MealyMachine<?, I, ?, O> ptaToModel(StrongBlueFringePTA<Void, O> pta) {
        CompactMealy<I, O> mealy = new CompactMealy<>(this.alphabet, pta.getNumRedStates());
        pta.toAutomaton(mealy, this.alphabet);
        return mealy;
    }

    protected RedBlueMerge<Void, O, StrongBlueFringePTAState<Void, O>> tryMerge(StrongBlueFringePTA<Void, O> pta, StrongBlueFringePTAState<Void, O> qr, StrongBlueFringePTAState<Void, O> qb) {
        return pta.tryMerge(qr, qb);
    }
}
