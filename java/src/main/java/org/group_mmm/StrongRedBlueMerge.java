/*
 * A large part of this code is copied & pasted from RedBlueMerge.java in LearnLib 0.14.0, http://www.learnlib.de/.
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
import net.automatalib.commons.smartcollections.ArrayStorage;
import net.automatalib.commons.util.Pair;

import javax.annotation.Nullable;
import java.lang.reflect.Field;
import java.util.*;
import java.util.logging.Logger;

import static java.lang.Integer.max;

public class StrongRedBlueMerge<SP, TP, S extends AbstractBlueFringePTAState<SP, TP, S>> extends RedBlueMerge<SP, TP, S> {
    protected int min_depth;

    //@ requires 0 <= min_depth;
    public StrongRedBlueMerge(AbstractBlueFringePTA<SP, TP, S> pta, S qr, S qb, int min_depth) {
        super(pta, qr, qb);
        assert min_depth >= 0 : "min_depth must be non-negative";
        this.min_depth = min_depth;
    }

    // We have to use reflection because these properties/methods are private/protected in the parent class
    int getAlphabetSize() {
        try {
            Field field = this.getClass().getSuperclass().getDeclaredField("alphabetSize");
            field.setAccessible(true);
            return (int) field.get(this);
        } catch (NoSuchFieldException | IllegalAccessException err) {
            Logger.getLogger(StrongRedBlueMerge.class.getName()).warning(err.getMessage());
            return -1;
        }
    }

    S getQr() throws NoSuchFieldException, IllegalAccessException {
        try {
            Field field = this.getClass().getSuperclass().getDeclaredField("qr");
            field.setAccessible(true);
            return (S) field.get(this);
        } catch (NoSuchFieldException | IllegalAccessException err) {
            Logger.getLogger(StrongRedBlueMerge.class.getName()).warning(err.getMessage());
            throw err;
        }
    }

    S getQb() throws NoSuchFieldException, IllegalAccessException {
        try {
            Field field = this.getClass().getSuperclass().getDeclaredField("qb");
            field.setAccessible(true);
            return (S) field.get(this);
        } catch (NoSuchFieldException | IllegalAccessException err) {
            Logger.getLogger(StrongRedBlueMerge.class.getName()).warning(err.getMessage());
            throw err;
        }
    }

    S getParent(S state) throws NoSuchFieldException, IllegalAccessException {
        try {
            Field field = state.getClass().getSuperclass().getDeclaredField("parent");
            field.setAccessible(true);
            return (S) field.get(state);
        } catch (NoSuchFieldException | IllegalAccessException err) {
            Logger.getLogger(StrongRedBlueMerge.class.getName()).warning(err.getMessage());
            throw err;
        }
    }

    int getParentInput(S state) throws NoSuchFieldException, IllegalAccessException {
        try {
            Field field = state.getClass().getSuperclass().getDeclaredField("parentInput");
            field.setAccessible(true);
            return (int) field.get(state);
        } catch (NoSuchFieldException | IllegalAccessException err) {
            Logger.getLogger(StrongRedBlueMerge.class.getName()).warning(err.getMessage());
            throw err;
        }
    }

    SP getProperty(S state) {
        try {
            Field field = state.getClass().getSuperclass().getSuperclass().getDeclaredField("property");
            field.setAccessible(true);
            return (SP) field.get(state);
        } catch (NoSuchFieldException | IllegalAccessException err) {
            Logger.getLogger(StrongRedBlueMerge.class.getName()).warning(err.getMessage());
            return null;
        }
    }

    ArrayStorage<TP> getTransPropertiesRefl(S state) {
        try {
            Field field = state.getClass().getSuperclass().getSuperclass().getDeclaredField("transProperties");
            field.setAccessible(true);
            return (ArrayStorage<TP>) field.get(state);
        } catch (NoSuchFieldException | IllegalAccessException err) {
            Logger.getLogger(StrongRedBlueMerge.class.getName()).warning(err.getMessage());
            return null;
        }
    }

    boolean getIsCopy(S state) throws NoSuchFieldException, IllegalAccessException {
        try {
            Field field = state.getClass().getSuperclass().getDeclaredField("isCopy");
            field.setAccessible(true);
            return (boolean) field.get(state);
        } catch (NoSuchFieldException | IllegalAccessException err) {
            Logger.getLogger(StrongRedBlueMerge.class.getName()).warning(err.getMessage());
            throw err;
        }
    }

    int getId(S state) {
        try {
            Field field = state.getClass().getSuperclass().getSuperclass().getDeclaredField("id");
            field.setAccessible(true);
            return (int) field.get(state);
        } catch (NoSuchFieldException | IllegalAccessException err) {
            Logger.getLogger(StrongRedBlueMerge.class.getName()).warning(err.getMessage());
            return -1;
        }
    }

    ArrayStorage<S> getSuccessors(S state) {
        try {
            Field field = state.getClass().getSuperclass().getSuperclass().getDeclaredField("successors");
            field.setAccessible(true);
            return (ArrayStorage<S>) field.get(state);
        } catch (NoSuchFieldException | IllegalAccessException err) {
            Logger.getLogger(StrongRedBlueMerge.class.getName()).warning(err.getMessage());
            return null;
        }
    }

    ArrayStorage<ArrayStorage<S>> getSuccMod() {
        try {
            Field field = this.getClass().getSuperclass().getDeclaredField("succMod");
            field.setAccessible(true);
            return (ArrayStorage<ArrayStorage<S>>) field.get(this);
        } catch (NoSuchFieldException | IllegalAccessException err) {
            Logger.getLogger(StrongRedBlueMerge.class.getName()).warning(err.getMessage());
            return null;
        }
    }

    ArrayStorage<ArrayStorage<TP>> getTransPropMod() {
        try {
            Field field = this.getClass().getSuperclass().getDeclaredField("transPropMod");
            field.setAccessible(true);
            return (ArrayStorage<ArrayStorage<TP>>) field.get(this);
        } catch (NoSuchFieldException | IllegalAccessException err) {
            Logger.getLogger(StrongRedBlueMerge.class.getName()).warning(err.getMessage());
            return null;
        }
    }

    ArrayStorage<SP> getPropMod() {
        try {
            Field field = this.getClass().getSuperclass().getDeclaredField("propMod");
            field.setAccessible(true);
            return (ArrayStorage<SP>) field.get(this);
        } catch (NoSuchFieldException | IllegalAccessException err) {
            Logger.getLogger(StrongRedBlueMerge.class.getName()).warning(err.getMessage());
            return null;
        }
    }

    void setMerged(boolean value) {
        try {
            Field field = this.getClass().getSuperclass().getDeclaredField("merged");
            field.setAccessible(true);
            field.set(this, value);
        } catch (NoSuchFieldException | IllegalAccessException err) {
            Logger.getLogger(StrongRedBlueMerge.class.getName()).warning(err.getMessage());
        }
    }


    void setProperty(S state, SP property) {
        try {
            Field field = state.getClass().getDeclaredField("property");
            field.setAccessible(true);
            field.set(state, property);
        } catch (NoSuchFieldException | IllegalAccessException err) {
            Logger.getLogger(StrongRedBlueMerge.class.getName()).warning(err.getMessage());
        }
    }

    public static <SP, TP, S extends AbstractBlueFringePTAState<SP, TP, S>> boolean
    notEnouchChildrenDepth(S q, int min_depth, int alphabetSize) {
        int maxDepth = 0;
        Deque<Pair<S, Integer>> stack = new ArrayDeque<>();
        stack.push(Pair.of(q, 0));
        Pair<S, Integer> currentPair;
        Set<S> visitedSet = new HashSet<>();
        while ((currentPair = stack.poll()) != null) {
            maxDepth = max(maxDepth, currentPair.getSecond());
            for (int i = 0; i < alphabetSize; i++) {
                S rSucc = currentPair.getFirst().getSuccessor(i);
                if (rSucc != null && !visitedSet.contains(rSucc)) {
                    visitedSet.add(rSucc);
                    stack.push(Pair.of(rSucc, currentPair.getSecond() + 1));
                    if (min_depth <= currentPair.getSecond() + 1) {
                        return false;
                    }
                }

            }
        }
        return min_depth > maxDepth;
    }

    public boolean merge() {
        int largest_depth = 0;
        try {
            S qr = getQr(), qb = getQb();
            this.setMerged(true);
            // Pruning when the depth of the children is too shallow
            if (notEnouchChildrenDepth(getQr(), min_depth, getAlphabetSize())) {
                return false;
            }

            if (!mergeRedProperties(qr, qb)) {
                return false;
            }

            updateRedTransition(getParent(qb), getParentInput(qb), qr);

            Deque<Pair<FoldRecord<S>, Integer>> stack = new ArrayDeque<>();
            stack.push(Pair.of(new FoldRecord<>(getQr(), getQb()), 0));

            Pair<FoldRecord<S>, Integer> currPair;
            while ((currPair = stack.peek()) != null) {
                FoldRecord<S> curr = currPair.getFirst();
                int depth = currPair.getSecond();
                int i = ++curr.i;

                if (i == getAlphabetSize()) {
                    stack.pop();
                    continue;
                }

                S q = curr.q;
                S r = curr.r;

                S rSucc = r.getSuccessor(i);
                if (rSucc != null) {
                    S qSucc = getSucc(q, i);
                    if (qSucc != null) {
                        if (qSucc.isRed()) {
                            if (!mergeRedProperties(qSucc, rSucc)) {
                                return false;
                            }
                        } else {
                            SP rSuccSP = getProperty(rSucc), qSuccSP = getProperty(qSucc);

                            SP newSP = null;
                            if (qSuccSP == null && rSuccSP != null) {
                                newSP = rSuccSP;
                            } else if (rSuccSP != null) { // && qSucc.property != null
                                if (!Objects.equals(qSuccSP, rSuccSP)) {
                                    return false;
                                }
                            }

                            ArrayStorage<TP> newTPs = null;
                            ArrayStorage<TP> rSuccTPs = getTransPropertiesRefl(rSucc);
                            ArrayStorage<TP> qSuccTPs = getTransPropertiesRefl(qSucc);

                            if (rSuccTPs != null) {
                                if (qSuccTPs != null) {
                                    ArrayStorage<TP> mergedTPs = mergeTransProperties(qSuccTPs, rSuccTPs);
                                    if (mergedTPs == null) {
                                        return false;
                                    } else if (mergedTPs != qSuccTPs) {
                                        newTPs = mergedTPs;
                                    }
                                } else {
                                    newTPs = rSuccTPs.clone();
                                }
                            }

                            if (newSP != null || newTPs != null) {
                                qSucc = cloneTopSucc(qSucc, i, stack, newTPs);
                                if (newSP != null) {
                                    setProperty(qSucc, newSP);
                                }
                            }
                        }

                        stack.push(Pair.of(new FoldRecord<>(qSucc, rSucc), depth + 1));
                        largest_depth = max(largest_depth, depth + 1);
                    } else {
                        if (q.isRed()) {
                            updateRedTransition(q, i, rSucc, r.getTransProperty(i));
                        } else {
                            q = cloneTop(q, stack);
                            assert getIsCopy(q);
                            q.setForeignSuccessor(i, rSucc, getAlphabetSize());
                        }
                    }
                }
            }
            // check the largest common depth
            return this.min_depth <= largest_depth;

        } catch (IllegalAccessException | NoSuchFieldException err) {
            Logger.getLogger(StrongRedBlueMerge.class.getName()).warning(err.getMessage());
            return false;
        }
    }

    private S cloneTopSucc(S succ, int i, Deque<Pair<FoldRecord<S>, Integer>> stack, @Nullable ArrayStorage<TP> newTPs) {
        S succClone = (newTPs != null) ? succ.copy(newTPs) : succ.copy();
        if (succClone == succ) {
            return succ;
        }
        Pair<FoldRecord<S>, Integer> peek = stack.peek();
        assert peek != null;
        S top = peek.getFirst().q;
        if (top.isRed()) {
            updateRedTransition(top, i, succClone);
        } else {
            S topClone = cloneTop(top, stack);
            topClone.setForeignSuccessor(i, succClone, getAlphabetSize());
        }
        return succClone;
    }

    private S cloneTop(S topState, Deque<Pair<FoldRecord<S>, Integer>> stack) {
        assert !topState.isRed();

        S topClone = topState.copy();
        if (topClone == topState) {
            return topState;
        }
        S currTgt = topClone;

        Iterator<Pair<FoldRecord<S>, Integer>> it = stack.iterator();
        FoldRecord<S> currRec = it.next().getFirst();
        assert currRec.q == topState;
        currRec.q = topClone;

        assert it.hasNext();
        currRec = it.next().getFirst();
        S currSrc = currRec.q;

        while (!currSrc.isRed()) {
            S currSrcClone = currSrc.copy();
            assert getSuccessors(currSrcClone) != null;
            getSuccessors(currSrcClone).set(currRec.i, currTgt);
            if (currSrcClone == currSrc) {
                return topClone; // we're done
            }
            currRec.q = currSrcClone;
            currTgt = currSrcClone;

            assert it.hasNext();
            currRec = it.next().getFirst();
            currSrc = currRec.q;
        }

        assert currSrc.isRed();
        updateRedTransition(currSrc, currRec.i, currTgt);

        return topClone;
    }

    private @Nullable
    ArrayStorage<TP> getTransProperties(S q) {
        if (q.isRed()) {
            int qId = getId(q);
            ArrayStorage<TP> props = getTransPropMod().get(qId);
            if (props != null) {
                return props;
            }
        }
        return getTransPropertiesRefl(q);
    }

    private SP getStateProperty(S q) {
        if (q.isRed()) {
            int qId = getId(q);
            SP prop = getPropMod().get(qId);
            if (prop != null) {
                return prop;
            }
        }
        return getProperty(q);
    }

    private @Nullable
    S getSucc(S q, int i) {
        if (q.isRed()) {
            int qId = getId(q);
            ArrayStorage<S> modSuccs = getSuccMod().get(qId);
            if (modSuccs != null) {
                return modSuccs.get(i);
            }
        }
        return q.getSuccessor(i);
    }

    private void updateRedTransition(S redSrc, int input, S tgt) {
        updateRedTransition(redSrc, input, tgt, null);
    }

    private void updateRedTransition(S redSrc, int input, S tgt, @Nullable TP transProp) {
        assert redSrc.isRed();

        int id = getId(redSrc);
        ArrayStorage<S> newSuccs = getSuccMod().get(id);
        if (newSuccs == null) {
            if (getSuccessors(redSrc) == null) {
                newSuccs = new ArrayStorage<>(getAlphabetSize());
            } else {
                newSuccs = getSuccessors(redSrc).clone();
            }
            getSuccMod().set(id, newSuccs);
        }
        newSuccs.set(input, tgt);
        if (transProp != null) {
            ArrayStorage<TP> newTransProps = getTransPropMod().get(id);
            if (newTransProps == null) {
                if (getTransPropertiesRefl(redSrc) == null) {
                    newTransProps = new ArrayStorage<>(getAlphabetSize());
                } else {
                    newTransProps = getTransPropertiesRefl(redSrc).clone();
                }
                getTransPropMod().set(id, newTransProps);
            }
            newTransProps.set(input, transProp);
        }
    }

    private boolean mergeRedProperties(S qr, S qb) {
        return mergeRedStateProperty(qr, qb) && mergeRedTransProperties(qr, qb);
    }

    private boolean mergeRedTransProperties(S qr, S qb) {
        assert qr.isRed();

        ArrayStorage<TP> qbProps = getTransPropertiesRefl(qb);
        if (qbProps == null) {
            return true;
        }
        ArrayStorage<TP> qrProps = getTransProperties(qr);
        ArrayStorage<TP> mergedProps = qbProps;
        if (qrProps != null) {
            mergedProps = mergeTransProperties(qrProps, qbProps);
            if (mergedProps == null) {
                return false;
            }
        }
        if (mergedProps != qrProps) {
            getTransPropMod().set(getId(qr), mergedProps);
        }
        return true;
    }

    private boolean mergeRedStateProperty(S qr, S qb) {
        assert qr.isRed();

        SP qbProp = getProperty(qb);
        if (qbProp == null) {
            return true;
        }
        SP qrProp = getStateProperty(qr);
        if (qrProp != null) {
            return Objects.equals(qbProp, qrProp);
        }
        getPropMod().set(getId(qr), qbProp);
        return true;
    }

    /**
     * Merges two non-null transition property arrays. The behavior of this method is as follows: <ul> <li>if {@code
     * tps1} subsumes {@code tps2}, then {@code tps1} is returned.</li> <li>otherwise, if {@code tps1} and {@code tps2}
     * can be merged, a new {@link ArrayStorage} containing the result of the merge is returned. <li>otherwise
     * (i.e., if no merge is possible), {@code null} is returned. </ul>
     */
    private @Nullable
    ArrayStorage<TP> mergeTransProperties(ArrayStorage<TP> tps1, ArrayStorage<TP> tps2) {
        int len = tps1.size();
        int i;

        ArrayStorage<TP> tps1OrCopy = tps1;

        for (i = 0; i < len; i++) {
            TP tp1 = tps1OrCopy.get(i);
            TP tp2 = tps2.get(i);
            if (tp2 != null) {
                if (tp1 != null) {
                    if (!Objects.equals(tp1, tp2)) {
                        return null;
                    }
                } else {
                    tps1OrCopy = tps1.clone();
                    tps1OrCopy.set(i++, tp2);
                    break;
                }
            }
        }

        for (; i < len; i++) {
            TP tp1 = tps1OrCopy.get(i);
            TP tp2 = tps2.get(i);
            if (tp2 != null) {
                if (tp1 != null) {
                    if (!Objects.equals(tp1, tp2)) {
                        return null;
                    }
                } else {
                    tps1OrCopy.set(i, tp2);
                }
            }
        }

        return tps1OrCopy;
    }

    static final class FoldRecord<S extends AbstractBlueFringePTAState<?, ?, S>> {

        public final S r;
        public S q;
        public int i = -1;

        FoldRecord(S q, S r) {
            this.q = q;
            this.r = r;
        }
    }
}
