package org.homework.util;

import java.io.IOException;
import java.io.OutputStream;
import java.util.Arrays;

public class Util {

    public static double[][] getDeepArrayCopy(final double[][] matrix) {
        return Arrays.stream(matrix).map(double[]::clone).toArray(double[][]::new);
    }

    public static int[] getIntArrayFromDoubleArray(final double[] x) {
        return Arrays.stream(x).mapToInt((val) -> new Double(val).intValue()).toArray();
    }

    public static double[] getDoubleArrayFromIntArray(final int[] x) {
        return Arrays.stream(x).mapToDouble((val) -> new Integer(val).doubleValue()).toArray();
    }

    /**
     * Close output stream
     *
     * @param fileOutputStream output stream created
     */
    public static void closeOutputStream(final OutputStream fileOutputStream) {
        if (fileOutputStream == null) return;

        try {
            fileOutputStream.close();
        } catch (final IOException e) {
            throw new RuntimeException(e);
        }
    }
}
