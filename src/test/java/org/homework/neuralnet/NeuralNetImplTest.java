package org.homework.neuralnet;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class NeuralNetImplTest {

  NeuralNetImpl neuralNet;

  @BeforeEach
  void setUp() {
    this.neuralNet = new NeuralNetImpl(2, 4, .2, .0, 1, 0, false);
  }

  @Test
  void TEST_TRAIN_XOR_RANDOM() {
    final double[][] x = this.getRandomInputVector(10, 2);
    final double[][] yHat = this.getRandomInputVector(10, 1);
    this.neuralNet.train(x, yHat);
    this.printNNTrainState(x, yHat, this.neuralNet.forward(x).toDoubleMatrix());
  }

  @Test
  void TEST_TRAIN_XOR_THEN_APPLY() {
    final double[][] x = new double[][] {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    final double[][] yHat = new double[][] {{0}, {1}, {1}, {0}};
    this.neuralNet.train(x, yHat);
    this.printNNTrainState(x, yHat, this.neuralNet.forward(x).toDoubleMatrix());
  }

  @Test
  void TEST_TRAIN_XOR_BIPOLAR() {
    this.neuralNet.setBipolar(true);
    final double[][] x = new double[][] {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
    final double[][] yHat = new double[][] {{-1}, {1}, {1}, {-1}};
    this.neuralNet.train(x, yHat);
    this.printNNTrainState(x, yHat, this.neuralNet.forward(x).toDoubleMatrix());
  }

  @Test
  void FORWARD_SHOULD_MAINTAIN_SHAPE() {
    final INDArray outputIndArray = this.neuralNet.forward(this.getRandomInputVector(1, 2));
    assertArrayEquals(new long[] {1, 1}, outputIndArray.shape());
  }

  @Test
  void LOSS_FUNCTION_CAL_AS_EXPECTED() {
    final double[][] randomOutputVector = this.getRandomInputVector(1, 1);
    final double[][] targetArray = this.getRandomInputVector(1, 1);
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

  private void printNNTrainState(
      final double[][] X, final double[][] target, final double[][] actual) {
    System.out.printf(
        "Input: \n %s \n Target y: \n %s \n Actual y: \n %s \n",
        Stream.of(X, target, actual).map(Arrays::deepToString).toArray());
  }
}
