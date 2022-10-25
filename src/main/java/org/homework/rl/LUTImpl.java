package org.homework.rl;

import lombok.Getter;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Random;

@Getter
public class LUTImpl implements LUTInterface {

  private static final Logger logger = LoggerFactory.getLogger(LUTImpl.class);
  private static final DataType LUT_DATA_TYPE = DataType.DOUBLE;
  private final int argNumInputs;
  private final long battleFieldWidth;
  private final long battleFieldHeight;
  private final double epsilon;
  private final int actionSize;
  private final int[] argVariableFloor;
  private final int[] argVariableCeiling;
  public INDArray qTable;

  public LUTImpl(
      final int argNumInputs,
      final long battleFieldWidth,
      final long battleFieldHeight,
      final int actionSize,
      final double epsilon,
      final int[] argVariableFloor,
      final int[] argVariableCeiling) {
    this.argNumInputs = argNumInputs;
    this.battleFieldWidth = battleFieldWidth;
    this.battleFieldHeight = battleFieldHeight;
    this.actionSize = actionSize;
    this.epsilon = epsilon;
    this.argVariableFloor = argVariableFloor;
    this.argVariableCeiling = argVariableCeiling;

    this.initialiseLUT();
  }

  @Override
  public void initialiseLUT() {
    this.qTable =
        Nd4j.rand(LUT_DATA_TYPE, this.battleFieldHeight, this.battleFieldHeight, this.actionSize);
  }

  @Override
  public int indexFor(final double[] X) {
    return 0;
  }

  /** Perform q learning on state action table */
  int chooseAction(final int... dimension) {
    if (Math.random() < this.epsilon) {
      // explore
      return this.chooseRandomAction();
    }

    return this.chooseGreedyAction(dimension);
  }

  int chooseGreedyAction(final int... dimension) {
    return 0;
  }

  int chooseRandomAction() {
    return new Random().nextInt(this.actionSize);
  }

  void setQValue(final double qValue, final int... dimension) {
    this.qTable.putScalar(dimension, qValue);
  }

  double getQValue(final int... dimension) {
    return this.qTable.getDouble(dimension);
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
}
