package org.homework.util;

import robocode.RobocodeFileWriter;

import java.io.File;

public class LogFile {
    public void writeToFile(final File fileToWrite, final double winRate, final int roundCount) {
        try {
            final RobocodeFileWriter fileWriter =
                    new RobocodeFileWriter(fileToWrite.getAbsolutePath(), true);
            fileWriter.write(roundCount + " " + Double.toString(winRate) + "\r\n");
            fileWriter.close();
        } catch (final Exception e) {
            System.out.println(e);
        }
    }
}
