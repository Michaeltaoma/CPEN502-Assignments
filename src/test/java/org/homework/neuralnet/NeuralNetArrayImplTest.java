package org.homework.neuralnet;

import org.homework.robot.model.Action;
import org.homework.robot.model.ImmutableState;
import org.homework.robot.model.State;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.homework.TestUtil.getRandomInputVector;
import static org.homework.TestUtil.printNNTrainState;
import static org.junit.jupiter.api.Assertions.assertEquals;

class NeuralNetArrayImplTest {

    NeuralNetArrayImpl neuralNetArray;

    int testArgNumInputs = 2;

    int testArgNumHidden = 4;

    int testArgNumOutputs = 1;

    @BeforeEach
    void setUp() {
        this.neuralNetArray =
                new NeuralNetArrayImpl(
                        this.testArgNumInputs,
                        this.testArgNumHidden,
                        this.testArgNumOutputs,
                        .4,
                        .0,
                        1,
                        0,
                        false);
    }

    @Test
    void TEST_TRAIN_XOR_RANDOM() {
        final double[][] x = getRandomInputVector(10, this.testArgNumInputs);
        final double[][] yHat = getRandomInputVector(10, this.testArgNumOutputs);
        this.neuralNetArray.train(x, yHat);
        printNNTrainState(x, yHat, this.neuralNetArray.batchForward(x).getData());
    }

    @Test
    void TEST_TRAIN_XOR_THEN_APPLY() {
        final double[][] x = new double[][] {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        final double[][] yHat = new double[][] {{0}, {1}, {1}, {0}};
        this.neuralNetArray.train(x, yHat);
        printNNTrainState(x, yHat, this.neuralNetArray.batchForward(x).getData());
    }

    @Test
    void TEST_TRAIN_XOR_BIPOLAR() {
        this.neuralNetArray.setBipolar(true);
        final double[][] x = new double[][] {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
        final double[][] yHat = new double[][] {{-1}, {1}, {1}, {-1}};
        this.neuralNetArray.train(x, yHat);
        printNNTrainState(x, yHat, this.neuralNetArray.batchForward(x).getData());
    }

    @Test
    void TEST_NN_FOR_DIFFERENT_DIMENSION() {
        final State mockState = ImmutableState.builder().build();
        this.neuralNetArray = new NeuralNetArrayImpl(mockState);
        assertEquals(
                Action.values().length,
                this.neuralNetArray
                        .forward(new double[1][mockState.getIndexedStateValue().length])
                        .getColNum());
    }
}
