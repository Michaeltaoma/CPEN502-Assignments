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
import java.util.Random;

@Getter
public class LUTImpl implements LUTInterface {

    private static final Logger logger = LoggerFactory.getLogger(LUTImpl.class);
    private static final DataType LUT_DATA_TYPE = DataType.DOUBLE;

    private final int myHPTypes;
    private final int enemyHPTypes;
    private final int distanceToEnemyTypes;
    private final int distanceToWallTypes;
    private final int actionSize;
    private final double epsilon;
    public INDArray qTable;

    public LUTImpl(
            final int myHPTypes,
            final int enemyHPTypes,
            final int distanceToEnemyTypes,
            final int distanceToWallTypes,
            final int actionSize,
            final double epsilon) {
        this.myHPTypes = myHPTypes;
        this.enemyHPTypes = enemyHPTypes;
        this.distanceToEnemyTypes = distanceToEnemyTypes;
        this.distanceToWallTypes = distanceToWallTypes;
        this.actionSize = actionSize;
        this.epsilon = epsilon;

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

    /** Perform q learning on state action table */
    public int chooseAction(final int[] dimension) {
        if (Math.random() < this.epsilon) {
            // explore
            return this.chooseRandomAction();
        }

        return this.chooseGreedyAction(dimension);
    }

    int chooseGreedyAction(final int[] dimension) {
        // given state, get the max action value from q table
        // example: action = np.argmax(Q_table[state[0], state[1],state[2],...])
        // -> find the best action based on Q(s,a') from Q(s)
        // so dimension should contain state info
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
