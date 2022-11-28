package org.homework.util;

import java.io.IOException;
import java.io.OutputStream;
import java.util.Arrays;

public class Util {

    public static double[][] getDeepArrayCopy(final double[][] matrix) {
        return Arrays.stream(matrix).map(double[]::clone).toArray(double[][]::new);
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
