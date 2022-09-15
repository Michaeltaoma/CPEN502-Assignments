package org.homework.neuralnet;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

public class NeuralNetImpl implements NeuralNetInterface{

    private static final Logger logger = LoggerFactory.getLogger(NeuralNetImpl.class);

    @Override
    public double outputFor(double[] X) {
        return 0;
    }

    @Override
    public double train(double[] X, double argValue) {
        return 0;
    }

    @Override
    public void save(File argFile) {

    }

    @Override
    public void load(String argFileName) throws IOException {

    }

    @Override
    public double sigmoid(double x) {
        return 0;
    }

    @Override
    public double customSigmoid(double x) {
        return 0;
    }

    @Override
    public void initializeWeights() {

    }

    @Override
    public void zeroWeights() {

    }
}
