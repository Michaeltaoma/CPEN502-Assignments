package org.homework.neuralnet;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.homework.neuralnet.matrix.Matrix;
import org.homework.robot.model.Action;
import org.homework.robot.model.State;
import org.homework.util.Util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

@Getter
@Setter
@NoArgsConstructor
public class NeuralNetArrayImpl implements NeuralNetInterface, Serializable {
    private static final long DEFAULT_EPOCH_CAPS = 1000;
    private static final int DEFAULT_ARG_NUM_INPUT_ROWS = 1;
    private static final int DEFAULT_HIDDEN_LAYER_NUM = 20;
    private static final int DEFAULT_PRINT_CYCLE = 100;
    private static final double DEFAULT_ERROR_THRESHOLD = 0.08;
    private static final double DEFAULT_RAND_RANGE_DIFFERENCE = .5;
    private static final double ALPHA = 0.1;
    private static final double GAMMA = 0.9;
    private static final double RANDOM_RATE = 0.1;
    private int argNumInputs;
    private int argNumOutputs;
    private int argNumHidden;
    private double argLearningRate;
    private double argMomentumTerm;
    private double sigmoidLowerBound;
    private double sigmoidUpperBound;
    private Matrix input;
    private Matrix inputToHiddenWeight;
    private Matrix deltaInputToHiddenWeight;
    private Matrix hiddenLayerBias;
    private Matrix deltaHiddenLayerBias;
    private Matrix hiddenOutput;
    private Matrix hiddenToOutputWeight;
    private Matrix deltaHiddenToOutputWeight;
    private Matrix output;
    private Matrix outputLayerBias;
    private Matrix deltaOutputLayerBias;
    private boolean isBipolar;

    public NeuralNetArrayImpl(
            final int argNumInputs,
            final int argNumHidden,
            final int argNumOutputs,
            final double argLearningRate,
            final double argMomentumTerm,
            final double sigmoidLowerBound,
            final double sigmoidUpperBound,
            final boolean isBipolar) {
        this.argNumInputs = argNumInputs;
        this.argNumHidden = argNumHidden;
        this.argNumOutputs = argNumOutputs;
        this.argLearningRate = argLearningRate;
        this.argMomentumTerm = argMomentumTerm;
        this.sigmoidLowerBound = sigmoidLowerBound;
        this.sigmoidUpperBound = sigmoidUpperBound;
        this.isBipolar = isBipolar;
        this.initialization();
    }

    public NeuralNetArrayImpl(
            final State state,
            final int argNumHidden,
            final double argLearningRate,
            final double argMomentumTerm,
            final boolean isBipolar) {
        this(
                state.getIndexedStateValue().length,
                argNumHidden,
                Action.values().length,
                argLearningRate,
                argMomentumTerm,
                1,
                0,
                isBipolar);
    }

    public NeuralNetArrayImpl(final State state) {
        this(state, 10, 0.1, 0.0, false);
    }

    @Override
    public double outputFor(final double[] X) {
        return this.forward(new double[][] {X}).data[0][0];
    }

    public double outputFor(final double[] X, final int index) {
        return this.forward(new double[][] {X}).data[0][index];
    }

    /**
     * One pass of forward feeding calculation
     *
     * @param X input vector
     * @return output vector as INDArray
     */
    public Matrix forward(final double[][] X) {
        this.input = new Matrix(X);
        // Hidden output = [x1, x2] * [[h11, h12, h13, h14], [h21, h22, h23, h24]] + [b1, b2, b3,
        // b4]
        this.hiddenOutput =
                this.sigmoidMatrix(
                        this.input.mmul(this.inputToHiddenWeight).add(this.hiddenLayerBias));
        this.output =
                this.sigmoidMatrix(
                        this.hiddenOutput
                                .mmul(this.hiddenToOutputWeight)
                                .add(this.outputLayerBias));
        return this.getOutput();
    }

    public Matrix batchForward(final double[][] X) {
        final int numSample = X.length;
        final double[][] res = new double[numSample][this.argNumOutputs];
        for (int i = 0; i < numSample; i++) {
            res[i] = this.forward(new double[][] {X[i]}).getData()[0];
        }
        return new Matrix(res);
    }

    @Override
    public double train(final double[] X, final double argValue) {
        return this.train(X, new double[] {argValue});
    }

    /**
     * The value that should be mapped to the given input vector. I.e. the desired correct output
     * vector for an input.
     *
     * @param X The input vector
     * @param argValue The output vector to learn
     * @return The error in the output for that input vector
     */
    public double train(final double[] X, final double[] argValue) {
        final Matrix output = this.forward(new double[][] {X});
        final Matrix targetOutput = new Matrix(new double[][] {argValue});
        final Matrix lossValue = this.loss(output, targetOutput);
        //        final double error = lossValue.sum(0).sumNumber().doubleValue();
        final double error = lossValue.sumValues();
        this.backpropagation(output, targetOutput);
        return error;
    }

    /**
     * Train batch of input data
     *
     * @param X Input data. Each row represent an input vector, X.length is the number of input
     *     data, X[0].length is the length of one input vector
     * @param argValues Target value. argValues[i] is the target value of X[i]
     */
    public void train(final double[][] X, final double[][] argValues) {
        final int numSample = X.length;
        int elapsedEpoch = 0;
        double totalError = 0, curEpochError = 0;
        do {
            for (int i = 0; i < numSample; i++) {
                curEpochError += this.train(X[i], argValues[i]);
            }

            curEpochError /= numSample;
            totalError += curEpochError;

            if (elapsedEpoch++ % DEFAULT_PRINT_CYCLE == 0)
                System.out.printf("Current Error: %f at Epoch %d%n", curEpochError, elapsedEpoch);

        } while (elapsedEpoch < DEFAULT_EPOCH_CAPS && curEpochError > DEFAULT_ERROR_THRESHOLD);

        System.out.printf(
                "NN trained for %d epochs, reached error per epoch = %.2f, best error: %.2f%n",
                elapsedEpoch, totalError / elapsedEpoch, curEpochError);
    }

    /**
     * Perform a backward propagation update for the weight matrix according to <a
     * href="http://courses.ece.ubc.ca/592/PDFfiles/Backpropagation_c.pdf">CPEN502 material</a>
     *
     * @param actualOutput The actual output of the neural network
     * @param targetOutput The target output of the neural network
     */
    private void backpropagation(final Matrix actualOutput, final Matrix targetOutput) {
        final Matrix actualTargetDiff = actualOutput.sub(targetOutput);

        // Calculate weight update for hiddenToOutputWeight
        final Matrix deltaOutputLayer =
                this.isBipolar
                        ? actualTargetDiff.mul(
                                (this.output.mul(-1).add(1)).mul((this.output.add(1))).mul(.5))
                        : actualTargetDiff.mul(this.output.mul(this.output.mul(-1).add(1)));

        this.deltaHiddenToOutputWeight =
                this.deltaHiddenToOutputWeight
                        .mul(this.argMomentumTerm)
                        .add(
                                this.hiddenOutput
                                        .transpose()
                                        .mul(this.argLearningRate)
                                        .mmul(deltaOutputLayer));

        // Calculate weight update for inputToHiddenWeight
        final Matrix deltaHiddenLayer =
                this.isBipolar
                        ? (this.hiddenOutput.mul(-1).add(1))
                                .mul(this.hiddenOutput.add(1))
                                .mul(.5)
                                .mul(deltaOutputLayer.mmul(this.hiddenToOutputWeight.transpose()))
                        : (this.hiddenOutput.mul(-1).add(1))
                                .mul(this.hiddenOutput)
                                .mul(deltaOutputLayer.mmul(this.hiddenToOutputWeight.transpose()));

        this.deltaInputToHiddenWeight =
                this.deltaInputToHiddenWeight
                        .mul(this.argMomentumTerm)
                        .add(
                                this.input
                                        .transpose()
                                        .mul(this.argLearningRate)
                                        .mmul(deltaHiddenLayer));

        // Perform the update
        this.hiddenToOutputWeight = this.hiddenToOutputWeight.sub(this.deltaHiddenToOutputWeight);
        this.inputToHiddenWeight = this.inputToHiddenWeight.sub(this.deltaInputToHiddenWeight);

        // Calculate bias update
        this.deltaHiddenLayerBias =
                this.deltaHiddenLayerBias
                        .mul(this.argMomentumTerm)
                        .add(deltaHiddenLayer.mul(this.argLearningRate));

        this.deltaOutputLayerBias =
                this.deltaOutputLayerBias
                        .mul(this.argMomentumTerm)
                        .add(deltaOutputLayer.mul(this.argLearningRate));

        this.hiddenLayerBias = this.hiddenLayerBias.sub(this.deltaHiddenLayerBias);
        this.outputLayerBias = this.outputLayerBias.sub(this.deltaOutputLayerBias);
    }

    /**
     * NEED TO DO
     *
     * @param actualOutput Actual output that the model produce
     * @param targetOutput Target output that the model need
     * @return Array of loss
     */
    public Matrix loss(final Matrix actualOutput, final Matrix targetOutput) {
        return actualOutput.sub(targetOutput).elementWiseOp(0, (a, b) -> Math.pow(a, 2)).mul(.5);
    }

    /**
     * Calculate sigmoid for every entry of the matrix
     *
     * @return Matrix after applying sigmoid
     */
    public Matrix sigmoidMatrix(final Matrix matrix) {
        return matrix.elementWiseOp(
                0, (a, b) -> this.isBipolar ? this.sigmoid(a) : this.customSigmoid(a));
    }

    @Override
    public double sigmoid(final double x) {
        return (2 / (1 + Math.pow(Math.E, (-1 * x)))) - 1;
    }

    @Override
    public double customSigmoid(final double x) {
        return 1 / (1 + Math.exp(-x));
    }

    private void initialization() {
        this.initializeWeights();
        this.initializeBias();
        this.initializeLayer();
    }

    @Override
    public void initializeWeights() {
        this.inputToHiddenWeight =
                Matrix.initRandMatrix(this.argNumInputs, this.argNumHidden)
                        .sub(DEFAULT_RAND_RANGE_DIFFERENCE);

        this.hiddenToOutputWeight =
                Matrix.initRandMatrix(this.argNumHidden, this.argNumOutputs)
                        .sub(DEFAULT_RAND_RANGE_DIFFERENCE);
    }

    @Override
    public void zeroWeights() {
        this.inputToHiddenWeight = Matrix.initZeroMatrix(this.argNumInputs, this.argNumHidden);

        this.hiddenToOutputWeight = Matrix.initZeroMatrix(this.argNumHidden, this.argNumOutputs);
    }

    /** Initialize bias value to 1.0 */
    private void initializeBias() {
        this.hiddenLayerBias =
                Matrix.initZeroMatrix(DEFAULT_ARG_NUM_INPUT_ROWS, this.argNumHidden).add(1.0);

        this.outputLayerBias =
                Matrix.initZeroMatrix(DEFAULT_ARG_NUM_INPUT_ROWS, this.argNumOutputs).add(1.0);
    }

    /** Initialize layer placeholder */
    private void initializeLayer() {
        this.input = Matrix.initRandMatrix(DEFAULT_ARG_NUM_INPUT_ROWS, this.argNumInputs);
        this.hiddenOutput = Matrix.initRandMatrix(DEFAULT_ARG_NUM_INPUT_ROWS, this.argNumHidden);
        this.output = Matrix.initRandMatrix(DEFAULT_ARG_NUM_INPUT_ROWS, this.argNumOutputs);
        this.deltaHiddenLayerBias =
                Matrix.initZeroMatrix(DEFAULT_ARG_NUM_INPUT_ROWS, this.argNumHidden);
        this.deltaOutputLayerBias =
                Matrix.initZeroMatrix(DEFAULT_ARG_NUM_INPUT_ROWS, this.argNumOutputs);
        this.deltaInputToHiddenWeight = Matrix.initZeroMatrix(this.argNumInputs, this.argNumHidden);
        this.deltaHiddenToOutputWeight =
                Matrix.initZeroMatrix(this.argNumHidden, this.argNumOutputs);
    }

    public void save(final String argFileName) {
        this.save(new File(argFileName));
    }

    @Override
    public void save(final File argFile) {
        try {
            final FileOutputStream fileOutputStream = new FileOutputStream(argFile, false);
            final PrintStream printStream = new PrintStream(fileOutputStream);

            final long inputToHiddenWeightRows = this.inputToHiddenWeight.rowNum;
            final long inputToHiddenWeightCols = this.inputToHiddenWeight.colNum;
            final long hiddenToOutputWeightRows = this.hiddenToOutputWeight.rowNum;
            final long hiddenToOutputWeightCols = this.hiddenToOutputWeight.colNum;
            printStream.println(inputToHiddenWeightRows);
            printStream.println(inputToHiddenWeightCols);
            printStream.println(hiddenToOutputWeightRows);
            printStream.println(hiddenToOutputWeightCols);

            for (int x = 0; x < inputToHiddenWeightRows; ++x) {
                for (int y = 0; y < inputToHiddenWeightCols; ++y) {
                    printStream.println(this.inputToHiddenWeight.data[x][y]);
                }
            }
            for (int x = 0; x < hiddenToOutputWeightRows; ++x) {
                for (int y = 0; y < hiddenToOutputWeightCols; ++y) {
                    printStream.println(this.hiddenToOutputWeight.data[x][y]);
                }
            }

            printStream.flush();
            printStream.close();
        } catch (final IOException error) {
            System.out.println("Failed to save the weights of a neural net.");
        }
    }

    public void load(final File argFileName) {}

    @Override
    public void load(final String argFileName) throws IOException {
        try {
            final BufferedReader bufferedReader = new BufferedReader(new FileReader(argFileName));
            final long inputToHiddenWeightRows = Long.parseLong(bufferedReader.readLine());
            final long inputToHiddenWeightCols = Long.parseLong(bufferedReader.readLine());
            final long hiddenToOutputWeightRows = Long.parseLong(bufferedReader.readLine());
            final long hiddenToOutputWeightCols = Long.parseLong(bufferedReader.readLine());
            if ((inputToHiddenWeightRows != this.inputToHiddenWeight.rowNum)
                    || (inputToHiddenWeightCols != this.inputToHiddenWeight.rowNum)) {
                System.out.println("wrong number of input neurons");
                bufferedReader.close();
                throw new IOException();
            }
            if ((hiddenToOutputWeightRows != this.hiddenToOutputWeight.rowNum)
                    || (hiddenToOutputWeightCols != this.hiddenToOutputWeight.colNum)) {
                System.out.println("wrong number of hidden neurons");
                bufferedReader.close();
                throw new IOException();
            }

            for (int x = 0; x < inputToHiddenWeightRows; ++x) {
                for (int y = 0; y < inputToHiddenWeightCols; ++y) {
                    this.inputToHiddenWeight.data[x][y] =
                            Double.parseDouble(bufferedReader.readLine());
                }
            }
            for (int x = 0; x < hiddenToOutputWeightRows; ++x) {
                for (int y = 0; y < hiddenToOutputWeightCols; ++y) {
                    this.hiddenToOutputWeight.data[x][y] =
                            Double.parseDouble(bufferedReader.readLine());
                }
            }

            bufferedReader.close();
        } catch (final IOException error) {
            System.out.println("Failed to open reader: " + error);
        }
    }

    public int chooseAction(final State currentState) {
        if (Math.random() < RANDOM_RATE) {
            return Action.values()[new Random().nextInt(Action.values().length)].ordinal();
        }

        double max = Double.NEGATIVE_INFINITY;
        Action action = Action.AHEAD;

        final double[] curQ =
                this.forward(
                                new double[][] {
                                    Util.getDoubleArrayFromIntArray(
                                            currentState.getIndexedStateValue())
                                })
                        .data[0];

        for (int i = 0; i < Action.values().length; i++) {
            if (max < curQ[i]) {
                action = Action.values()[i];
                max = curQ[i];
            }
        }

        return action.ordinal();
    }

    public void qTrain(
            final double reward,
            final State prevState,
            final Action prevAction,
            final State curState) {

        final double[] prevQ =
                this.forward(
                                new double[][] {
                                    Util.getDoubleArrayFromIntArray(
                                            prevState.getIndexedStateValue())
                                })
                        .data[0];

        final double curQ =
                Arrays.stream(
                                this.forward(
                                                new double[][] {
                                                    Util.getDoubleArrayFromIntArray(
                                                            curState.getIndexedStateValue())
                                                })
                                        .data[0])
                        .max()
                        .orElse(0);

        final double loss = ALPHA * (reward + GAMMA * curQ - prevQ[prevAction.ordinal()]);

        final double updatedQ = prevQ[prevAction.ordinal()] + loss;

        final double[] targetQ = prevQ.clone();
        targetQ[prevAction.ordinal()] = updatedQ;

        this.backpropagation(
                new Matrix(new double[][] {prevQ}), new Matrix(new double[][] {targetQ}));
    }

    public void sarsa(
            final double reward,
            final State prevState,
            final Action prevAction,
            final State curState,
            final Action curAction) {

        final double[] prevQ =
                this.forward(
                                new double[][] {
                                    Util.getDoubleArrayFromIntArray(
                                            prevState.getIndexedStateValue())
                                })
                        .data[0];

        final double[] curQ =
                this.forward(
                                new double[][] {
                                    Util.getDoubleArrayFromIntArray(curState.getIndexedStateValue())
                                })
                        .data[0];

        final double[] targetQ = prevQ.clone();
        targetQ[prevAction.ordinal()] =
                prevQ[prevAction.ordinal()]
                        + ALPHA
                                * (reward
                                        + GAMMA * curQ[curAction.ordinal()]
                                        - prevQ[prevAction.ordinal()]);

        this.backpropagation(
                new Matrix(new double[][] {prevQ}), new Matrix(new double[][] {targetQ}));
    }
}
