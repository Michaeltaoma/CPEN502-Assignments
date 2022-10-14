package org.homework.rl;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

public class LUTImpl implements LUTInterface {

  private static final Logger logger = LoggerFactory.getLogger(LUTImpl.class);
  private final int argNumInputs;
  private final int[] argVariableFloor;
  private final int[] argVariableCeiling;

  public LUTImpl(
      final int argNumInputs, final int[] argVariableFloor, final int[] argVariableCeiling) {
    this.argNumInputs = argNumInputs;
    this.argVariableFloor = argVariableFloor;
    this.argVariableCeiling = argVariableCeiling;
  }

  @Override
  public double outputFor(final double[] X) {
    return 0;
  }

  @Override
  public double train(final double[] X, final double argValue) {
    return 0;
  }

  @Override
  public void save(final File argFile) {}

  @Override
  public void load(final String argFileName) throws IOException {}

  @Override
  public void initialiseLUT() {}

  @Override
  public int indexFor(final double[] X) {
    return 0;
  }
}
