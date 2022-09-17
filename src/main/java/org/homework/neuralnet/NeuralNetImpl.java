package org.homework.neuralnet;

import lombok.Getter;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;

@Getter
public class NeuralNetImpl implements NeuralNetInterface, Serializable {
    private static final Logger logger = LoggerFactory.getLogger(NeuralNetImpl.class);
    private static final DataType NEURAL_NET_DATA_TYPE = DataType.DOUBLE;
    private static final double DEFAULT_RAND_RANGE_DIFFERENCE = .5;
    private static final int DEFAULT_ARG_NUM_INPUT_ROWS = 1;
    private static final int DEFAULT_ARG_NUM_OUTPUTS_COLS = 1;
    private final int argNumInputs;
    private final int argNumHidden;
    private final double argLearningRate;
    private final double argMomentumTerm;
    private final double sigmoidLowerBound;
    private final double sigmoidUpperBound;
    private INDArray inputToHiddenWeight;
    private INDArray hiddenLayerBias;
    private INDArray hiddenToOutputWeight;
    private INDArray outputLayerBias;

    public NeuralNetImpl(final int argNumInputs, final int argNumHidden, final double argLearningRate, final double argMomentumTerm, final double sigmoidLowerBound, final double sigmoidUpperBound) {
        this.argNumInputs = argNumInputs;
        this.argNumHidden = argNumHidden;
        this.argLearningRate = argLearningRate;
        this.argMomentumTerm = argMomentumTerm;
        this.sigmoidLowerBound = sigmoidLowerBound;
        this.sigmoidUpperBound = sigmoidUpperBound;
        this.initializeWeights();
        this.initializeBias();
    }

    @Override
    public double outputFor(final double[] X) {
        return this.forward(new double[][]{X}).toDoubleMatrix()[0][0];
    }

    /**
     * One pass of forward feeding calculation
     *
     * @param X input vector
     * @return output vector as INDArray
     */
    public INDArray forward(final double[][] X) {
        final INDArray inputIndArray = Nd4j.create(X);
        // Hidden output = [x1, x2] * [[h11, h12, h13, h14], [h21, h22, h23, h24]] + [b1, b2, b3, b4]
        final INDArray outputAfterHidden = Transforms.sigmoid(inputIndArray.mmul(this.inputToHiddenWeight).addi(this.hiddenLayerBias));
        return Transforms.sigmoid(outputAfterHidden.mmul(this.hiddenToOutputWeight).addi(this.outputLayerBias));
    }

    @Override
    public double train(final double[] X, final double argValue) {
        return 0;
    }

    @Override
    public void save(final File argFile) {

    }

    @Override
    public void load(final String argFileName) throws IOException {

    }

    @Override
    public double sigmoid(final double x) {
        return 2 / (1 + Math.pow(Math.E, (-1 * x))) - 1;
    }

    @Override
    public double customSigmoid(final double x) {
        return (this.sigmoidUpperBound - this.sigmoidLowerBound) / (1 + Math.pow(Math.E, (-1 * x))) - this.sigmoidLowerBound;
    }

    @Override
    public void initializeWeights() {
        this.inputToHiddenWeight = Nd4j.rand(NEURAL_NET_DATA_TYPE, this.argNumInputs, this.argNumHidden).subi(DEFAULT_RAND_RANGE_DIFFERENCE);
        this.hiddenToOutputWeight = Nd4j.rand(NEURAL_NET_DATA_TYPE, this.argNumHidden, DEFAULT_ARG_NUM_OUTPUTS_COLS).subi(DEFAULT_RAND_RANGE_DIFFERENCE);
    }

    @Override
    public void zeroWeights() {
        this.inputToHiddenWeight = Nd4j.zeros(NEURAL_NET_DATA_TYPE, this.argNumInputs, this.argNumHidden);
        this.hiddenToOutputWeight = Nd4j.zeros(NEURAL_NET_DATA_TYPE, this.argNumHidden, DEFAULT_ARG_NUM_OUTPUTS_COLS);
    }

    public void initializeBias() {
        this.hiddenLayerBias = Nd4j.ones(NEURAL_NET_DATA_TYPE, DEFAULT_ARG_NUM_INPUT_ROWS, this.argNumHidden);
        this.outputLayerBias = Nd4j.ones(NEURAL_NET_DATA_TYPE, DEFAULT_ARG_NUM_INPUT_ROWS, DEFAULT_ARG_NUM_OUTPUTS_COLS);
    }
}
