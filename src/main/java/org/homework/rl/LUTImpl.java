package org.homework.rl;

import lombok.Getter;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

@Getter
public class LUTImpl implements LUTInterface {

    private static final Logger logger = LoggerFactory.getLogger(LUTImpl.class);
    private static final DataType LUT_DATA_TYPE = DataType.DOUBLE;

    private static final double learningRate = 0.1;
    private static final double discountFactor = 0.9;
    private static final double epsilon = 0.9;

    private final int myHPTypes;
    private final int enemyHPTypes;
    private final int distanceToEnemyTypes;
    private final int distanceToWallTypes;
    private final int actionSize;
    public INDArray qTable;

    public LUTImpl(
            final int myHPTypes,
            final int enemyHPTypes,
            final int distanceToEnemyTypes,
            final int distanceToWallTypes,
            final int actionSize) {
        this.myHPTypes = myHPTypes;
        this.enemyHPTypes = enemyHPTypes;
        this.distanceToEnemyTypes = distanceToEnemyTypes;
        this.distanceToWallTypes = distanceToWallTypes;
        this.actionSize = actionSize;

        this.initialiseLUT();
    }

    @Override
    public void initialiseLUT() {
        this.qTable =
                Nd4j.rand(
                        LUT_DATA_TYPE,
                        this.myHPTypes,
                        this.enemyHPTypes,
                        this.distanceToEnemyTypes,
                        this.distanceToWallTypes,
                        this.actionSize);
    }

    @Override
    public int indexFor(final double[] X) {
        return 0;
    }

    /**
     * most time we will find the greedy action, but also do some exploration
     * @param dimension an array contains states info which will be used later for choosing best action
     * @return the index of the chosen action
     */
    public int chooseAction(final int[] dimension) {
        if (Math.random() < epsilon) {
            // explore
            return this.chooseRandomAction();
        }
        return this.chooseGreedyAction(dimension);
    }

    /**
     * given state, get the max action value from q table
     * @param dimension an array contains states info, so that we can find the best action based on Q(s,a') from Q(s)
     * @return the index of the best action
     */
    int chooseGreedyAction(final int[] dimension) {
        INDArray qValues = this.qTable.get(NDArrayIndex.point(dimension[0]));
        for (int i = 1; i < dimension.length; ++i) {
            qValues = qValues.get(NDArrayIndex.point(dimension[i]));
        }
        return qValues.argMax().getInt();
    }

    int chooseRandomAction() {
        return new Random().nextInt(this.actionSize);
    }

    void setQValue(final double qValue, final int[] dimension) {
        this.qTable.putScalar(dimension, qValue);
    }

    double getQValue(final int[] dimension) {
        return this.qTable.getDouble(dimension);
    }

    /**
     *
     * @param prevDimension an array contains previous states and action info to get previous Q value Q(s',a')
     * @param curDimension an array contains current states and action info to get current Q value Q(s,a)
     * @param reward an integer represents reward
     * @param isOnPolicy if true, then use Sarsa. Otherwise, use Q learning
     */
    public void computeQValue(int[] prevDimension, int[] curDimension, double reward, boolean isOnPolicy) {
        double prevQValue = getQValue(prevDimension);
        double curQValue = getQValue(curDimension);
        if (isOnPolicy) {
            // Sarsa
            setQValue(prevQValue + learningRate * (reward + discountFactor * curQValue - prevQValue), prevDimension);
        } else {
            // Q learning
            curDimension[curDimension.length-1] = chooseGreedyAction(Arrays.copyOf(curDimension, curDimension.length-1));
            double maxQValue = getQValue(curDimension);
            setQValue(prevQValue + learningRate * (reward + discountFactor * maxQValue - prevQValue), prevDimension);
        }
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
