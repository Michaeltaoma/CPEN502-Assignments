package org.homework.util;

import robocode.RobocodeFileWriter;

import java.io.File;
import java.util.List;

public class LogFileUtil {
    public void writeToFile(
            final File fileToWrite,
            final double winRate,
            final int roundCount,
            final boolean append) {
        try {
            final RobocodeFileWriter fileWriter =
                    new RobocodeFileWriter(fileToWrite.getAbsolutePath(), append);
            fileWriter.write(roundCount + " " + winRate + "\r\n");
            fileWriter.close();
        } catch (final Exception e) {
            System.out.println(e);
        }
    }

    public void writeRMSEToFile(
            final File fileToWrite,
            final List<Double> rMse,
            final int logCycle,
            final boolean append) {
        try {
            final RobocodeFileWriter fileWriter =
                    new RobocodeFileWriter(fileToWrite.getAbsolutePath(), append);
            for (int i = 0; i < rMse.size(); i++) {
                if (i % logCycle == 0) fileWriter.write(rMse.get(i) + "\r\n");
            }
            fileWriter.close();
        } catch (final Exception e) {
            System.out.println(e);
        }
    }
}
