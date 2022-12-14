package org.homework.neuralnet;

import org.homework.rl.LUTImpl;
import org.homework.robot.model.Action;
import org.homework.robot.model.ImmutableState;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;

class EnsembleNeuralNetTest {

    @BeforeEach
    void setUp() {}

    @Test
    void QTRAIN_WORK() {
        final EnsembleNeuralNet mockEnsembleNeuralNet =
                new EnsembleNeuralNet(Action.values().length);

        mockEnsembleNeuralNet.qTrain(
                0.1,
                ImmutableState.builder().build(),
                Action.AHEAD,
                ImmutableState.builder().build());
    }

    @Test
    void ONE_NN_FOR_EACH_ACTION() throws IOException {
        // lut q table gained from AI robot
        final String offlineTrainingDate = "robot-log/AIRobot-crazy-robot.txt";
        final LUTImpl lut = new LUTImpl(ImmutableState.builder().build());
        lut.load(offlineTrainingDate);

        final EnsembleNeuralNet ensembleNeuralNet = new EnsembleNeuralNet(Action.values().length);

        ensembleNeuralNet.train(lut.qTable);

        for (int i = 0; i < ensembleNeuralNet.neuralNetArrays.length; i++) {
            System.out.printf(
                    "%s error log: %s%n",
                    Action.values()[i].name(), ensembleNeuralNet.neuralNetArrays[i].getErrorLog());
        }
    }

    @Test
    void ONE_NN_FOR_EACH_ACTION_AGGREGATE_TRAIN() throws IOException {
        // lut q table gained from AI robot
        final String offlineTrainingDate = "robot-log/AIRobot-crazy-robot.txt";
        final LUTImpl lut = new LUTImpl(ImmutableState.builder().build());
        lut.load(offlineTrainingDate);

        final EnsembleNeuralNet ensembleNeuralNet = new EnsembleNeuralNet(Action.values().length);

        ensembleNeuralNet.train(lut.qTable, 10000, 0.01);

        System.out.println(ensembleNeuralNet.getErrorLog());
    }
}
