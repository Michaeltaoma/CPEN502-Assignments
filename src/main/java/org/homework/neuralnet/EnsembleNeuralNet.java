package org.homework.neuralnet;

import lombok.Getter;
import lombok.NoArgsConstructor;
import org.homework.neuralnet.matrix.Matrix;
import org.homework.robot.model.Action;
import org.homework.robot.model.ImmutableState;
import org.homework.robot.model.State;
import org.homework.util.Util;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Map;
import java.util.Random;

@NoArgsConstructor
@Getter
public class EnsembleNeuralNet {
    private static final double ALPHA = 0.1;
    private static final double GAMMA = 0.9;
    private static final double RANDOM_RATE = 0.4;
    public NeuralNetArrayImpl[] neuralNetArrays;
    public int actionSize;

    public EnsembleNeuralNet(final int actionSize) {
        this.actionSize = actionSize;

        this.neuralNetArrays = new NeuralNetArrayImpl[actionSize];

        for (int i = 0; i < actionSize; i++) {
            this.neuralNetArrays[i] =
                    new NeuralNetArrayImpl(
                            Action.values()[i].name(), ImmutableState.builder().build());
        }
    }

    double[] forward(final State state) {
        final double[] res = new double[this.neuralNetArrays.length];
        for (int i = 0; i < res.length; i++) {
            res[i] = this.neuralNetArrays[i].forward(state).data[0][0];
        }
        return res;
    }

    void train(final Map<State, double[]> qTable) {
        final double[][][] trainX =
                new double[this.neuralNetArrays.length][qTable.size()]
                        [ImmutableState.builder().build().getIndexedStateValue().length];

        final double[][][] trainY = new double[this.neuralNetArrays.length][qTable.size()][1];

        int curCount = 0;

        for (final Map.Entry<State, double[]> entry : qTable.entrySet()) {
            for (int i = 0; i < this.neuralNetArrays.length; i++) {
                trainX[i][curCount] =
                        Util.getDoubleArrayFromIntArray(entry.getKey().getIndexedStateValue());
                trainY[i][curCount] = new double[] {entry.getValue()[i]};
            }
            curCount++;
        }

        for (int i = 0; i < this.neuralNetArrays.length; i++) {
            this.neuralNetArrays[i].train(trainX[i], trainY[i]);
        }
    }

    public int chooseAction(final State currentState) {
        if (Math.random() < RANDOM_RATE) {
            return Action.values()[new Random().nextInt(Action.values().length)].ordinal();
        }

        double max = Double.NEGATIVE_INFINITY;
        Action action = Action.AHEAD;

        final double[] curQ = this.forward(currentState);

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

        final double[] prevQ = this.forward(prevState);

        final double curQ = Arrays.stream(this.forward(curState)).max().orElse(0);

        final double loss = ALPHA * (reward + GAMMA * curQ - prevQ[prevAction.ordinal()]);

        final double updatedQ = prevQ[prevAction.ordinal()] + loss;

        this.neuralNetArrays[prevAction.ordinal()].backpropagation(
                new Matrix(new double[][] {new double[] {prevQ[prevAction.ordinal()]}}),
                new Matrix(new double[][] {new double[] {updatedQ}}));
    }

    public void load(final File dataFile) throws IOException {}
}
