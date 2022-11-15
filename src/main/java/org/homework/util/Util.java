package org.homework.util;

import java.io.IOException;
import java.io.OutputStream;

public class Util {

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
