package org.homework;

import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.stream.Stream;

public class TestUtil {
    /**
     * get random input vector
     *
     * @param nRows num rows for input vector
     * @param nCols num cols for input vector
     * @return input vector
     */
    public static double[][] getRandomInputVector(final int nRows, final int nCols) {
        return Nd4j.rand(nRows, nCols).toDoubleMatrix();
    }

    public static void printNNTrainState(
            final double[][] X, final double[][] target, final double[][] actual) {
        System.out.printf(
                "Input: \n %s \n Target y: \n %s \n Actual y: \n %s \n",
                Stream.of(X, target, actual).map(Arrays::deepToString).toArray());
    }
}
