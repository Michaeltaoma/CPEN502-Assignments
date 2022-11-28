package org.homework.neuralnet;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;

import static org.homework.TestUtil.getRandomInputVector;
import static org.homework.TestUtil.printNNTrainState;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class NeuralNetImplTest {

    NeuralNetImpl neuralNet;
    int testArgNumInputs = 2;
    int testArgNumHidden = 10;
    int testArgNumOutputs = 1;

    @BeforeEach
    void setUp() {
        this.neuralNet =
                new NeuralNetImpl(
                        this.testArgNumInputs,
                        this.testArgNumHidden,
                        this.testArgNumOutputs,
                        .2,
                        .0,
                        1,
                        0,
                        false);
    }

    @Test
    void TEST_TRAIN_XOR_RANDOM() {
        final double[][] x = getRandomInputVector(10, this.testArgNumInputs);
        final double[][] yHat = getRandomInputVector(10, this.testArgNumOutputs);
        this.neuralNet.train(x, yHat);
        printNNTrainState(x, yHat, this.neuralNet.forward(x).toDoubleMatrix());
    }

    @Test
    void TEST_TRAIN_XOR_THEN_APPLY() {
        final double[][] x = new double[][] {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        final double[][] yHat = new double[][] {{0}, {1}, {1}, {0}};
        this.neuralNet.train(x, yHat);
        printNNTrainState(x, yHat, this.neuralNet.forward(x).toDoubleMatrix());
    }

    @Test
    void TEST_TRAIN_XOR_BIPOLAR() {
        this.neuralNet.setBipolar(true);
        final double[][] x = new double[][] {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
        final double[][] yHat = new double[][] {{-1}, {1}, {1}, {-1}};
        this.neuralNet.train(x, yHat);
        printNNTrainState(x, yHat, this.neuralNet.forward(x).toDoubleMatrix());
    }

    @Test
    void FORWARD_SHOULD_MAINTAIN_SHAPE() {
        final INDArray outputIndArray = this.neuralNet.forward(getRandomInputVector(1, 2));
        assertArrayEquals(new long[] {1, 1}, outputIndArray.shape());
    }

    @Test
    void LOSS_FUNCTION_CAL_AS_EXPECTED() {
        final double[][] randomOutputVector = getRandomInputVector(1, 1);
        final double[][] targetArray = getRandomInputVector(1, 1);
        final double[][] manualLoss = new double[1][1];
        manualLoss[0][0] = Math.pow(randomOutputVector[0][0] - targetArray[0][0], 2) / 2;
        assertArrayEquals(
                manualLoss,
                this.neuralNet
                        .loss(Nd4j.create(randomOutputVector), Nd4j.create(targetArray))
                        .toDoubleMatrix());
    }

    @Test
    void SAVE_AND_LOAD_WEIGHTS() throws IOException {
        final INDArray inputToHiddenWeight = this.neuralNet.getInputToHiddenWeight();
        final INDArray hiddenToOutputWeight = this.neuralNet.getHiddenToOutputWeight();

        final String filename = "./weights";
        final File file = new File(filename);
        this.neuralNet.save(file);
        this.neuralNet.load(filename);

        assertEquals(inputToHiddenWeight, this.neuralNet.getInputToHiddenWeight());
        assertEquals(hiddenToOutputWeight, this.neuralNet.getHiddenToOutputWeight());
    }
}
