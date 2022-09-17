package org.homework.neuralnet;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class NeuralNetImplTest {


    @BeforeEach
    void setUp() {
    }

    @Test
    void FORWARD_SHOULD_MAINTAIN_SHAPE() {
        final NeuralNetImpl neuralNet = new NeuralNetImpl(5, 10, .5, .2, 1, 0);
        final INDArray outputIndArray = neuralNet.forward(this.getRandomInputVector(1, 5));
        assertArrayEquals(new long[]{1, 1}, outputIndArray.shape());
    }

    /**
     * get random input vector
     *
     * @param nRows num rows for input vector
     * @param nCols num cols for input vector
     * @return input vector
     */
    private double[][] getRandomInputVector(final int nRows, final int nCols) {
        return Nd4j.rand(nRows, nCols).toDoubleMatrix();
    }
}