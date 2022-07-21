package org.group_mmm;

import com.pholser.junit.quickcheck.Property;
import com.pholser.junit.quickcheck.generator.InRange;
import com.pholser.junit.quickcheck.runner.JUnitQuickcheck;
import de.learnlib.algorithms.rpni.BlueFringeRPNIMealy;
import net.automatalib.automata.transducers.MealyMachine;
import net.automatalib.serialization.dot.GraphDOT;
import net.automatalib.words.Word;
import net.automatalib.words.impl.Alphabets;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.IOException;
import java.io.StringWriter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

@RunWith(JUnitQuickcheck.class)
public class StrongBlueFringeRPNIMealyTest {
    @Property
    public void computeModel(HashMap<List<@InRange(min = "0", max = "3") Integer>, @InRange(min = "0", max = "3") Integer> samples) throws IOException {
        List<Integer> alphabet = Arrays.stream(new int[]{0, 1, 2, 3}).boxed().collect(Collectors.toList());
        if (samples.isEmpty()) {
            return;
        }
        StrongBlueFringeRPNIMealy<Integer, Integer> learner = new StrongBlueFringeRPNIMealy<>(Alphabets.integers(0, 3), 0, 0);
        learner.setDeterministic(true);
        BlueFringeRPNIMealy<Integer, Integer> originalLearner = new BlueFringeRPNIMealy<>(Alphabets.integers(0, 3));
        originalLearner.setDeterministic(true);
        for (List<Integer> input : samples.keySet()) {
            if (input.isEmpty()) {
                if (samples.size() == 1) {
                    return;
                } else {
                    continue;
                }
            }
            Word<Integer> inputWord = Word.fromList(input);
            learner.addSample(inputWord, Word.fromLetter(samples.get(input)));
            originalLearner.addSample(inputWord, Word.fromLetter(samples.get(input)));
        }
        MealyMachine<?, Integer, ?, Integer> model;
        try {
            model = learner.computeModel();
        } catch (IllegalStateException ignored) {
            assertThrows(IllegalStateException.class, originalLearner::computeModel);
            return;
        }
        MealyMachine<?, Integer, ?, Integer> originalModel = originalLearner.computeModel();
        StringWriter modelStringWriter = new StringWriter(), originalStringWriter = new StringWriter();
        GraphDOT.write(model, alphabet, modelStringWriter);
        GraphDOT.write(originalModel, alphabet, originalStringWriter);
        assertEquals(modelStringWriter.toString(), originalStringWriter.toString());
    }

    @Test
    public void computeModelFixed() {
        Map<List<Integer>, Integer> samples = new HashMap<>();
        samples.put(Arrays.stream(new int[]{3, 2, 0, 1, 1, 1, 3, 2, 1, 2, 3, 2, 0, 1, 0, 3, 0, 3, 2, 0, 3, 2}).boxed().collect(Collectors.toList()), 2);
        samples.put(Arrays.stream(new int[]{1, 0, 1, 0, 1, 3, 2, 3, 1, 1, 1, 0, 2, 1, 1, 1, 1, 0, 1, 1, 0}).boxed().collect(Collectors.toList()), 2);
        samples.put(Arrays.stream(new int[]{2, 0, 1}).boxed().collect(Collectors.toList()), 0);
        samples.put(Arrays.stream(new int[]{2, 1, 0, 3, 1, 2}).boxed().collect(Collectors.toList()), 1);
        samples.put(Arrays.stream(new int[]{0, 3, 1, 0, 0, 3, 2, 3, 2, 2, 3, 1, 2, 3, 0, 1}).boxed().collect(Collectors.toList()), 3);
        samples.put(Arrays.stream(new int[]{1, 1, 2, 3, 3, 0, 0, 0, 1, 1, 3, 0, 3, 0, 2, 0}).boxed().collect(Collectors.toList()), 0);
        samples.put(Arrays.stream(new int[]{2, 2, 1, 0, 1, 3}).boxed().collect(Collectors.toList()), 3);
        samples.put(Arrays.stream(new int[]{0, 2, 3, 3, 2, 2, 1, 1, 2}).boxed().collect(Collectors.toList()), 2);

        //StrongBlueFringeRPNIMealy<Integer, Integer> learner = new StrongBlueFringeRPNIMealy<>(Alphabets.integers(0, 3), 0, 0);
        BlueFringeRPNIMealy<Integer, Integer> learner = new BlueFringeRPNIMealy<>(Alphabets.integers(0, 3));
        learner.setDeterministic(true);
        for (List<Integer> input : samples.keySet()) {
            if (input.isEmpty()) {
                if (samples.size() == 1) {
                    return;
                } else {
                    continue;
                }
            }
            Word<Integer> inputWord = Word.fromList(input);
            learner.addSample(inputWord, Word.fromLetter(samples.get(input)));
        }
        try {
            learner.computeModel();
        } catch (IllegalStateException ignored) {

        }
    }
}