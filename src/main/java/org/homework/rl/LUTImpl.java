package org.homework.rl;

import lombok.Getter;
import org.homework.robot.model.Action;
import org.homework.robot.model.ImmutableState;
import org.homework.robot.model.State;
import org.homework.robot.model.StateName;
import robocode.RobocodeFileOutputStream;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import static org.homework.robot.model.StateName.StateType.DISTANCE_TO_ENEMY;
import static org.homework.robot.model.StateName.StateType.DISTANCE_TO_WALL;
import static org.homework.robot.model.StateName.StateType.ENEMY_HP;
import static org.homework.robot.model.StateName.StateType.MY_HP;

@Getter
public class LUTImpl implements LUTInterface {
    private static final double learningRate = 0.1;
    private static final double discountFactor = 0.9;
    private static final double epsilon = 0.9;
    private final int myHPTypes;
    private final int enemyHPTypes;
    private final int distanceToEnemyTypes;
    private final int distanceToWallTypes;
    private final int actionSize;
    public Map<State, double[]> qTable;

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

    public LUTImpl(final State state) {
        this.myHPTypes = this.getStateDimension(state, MY_HP);
        this.enemyHPTypes = this.getStateDimension(state, ENEMY_HP);
        this.distanceToEnemyTypes = this.getStateDimension(state, DISTANCE_TO_ENEMY);
        this.distanceToWallTypes = this.getStateDimension(state, DISTANCE_TO_WALL);
        this.actionSize = Action.values().length;
        this.initialiseLUT();
    }

    @Override
    public void initialiseLUT() {
        this.qTable = new HashMap<>();
    }

    @Override
    public int indexFor(final double[] X) {
        return 0;
    }

    /**
     * most time we will find the greedy action, but also do some exploration
     *
     * @param state a State object represent the current state
     * @return the index of the chosen action
     */
    public int chooseAction(final State state) {
        if (Math.random() < epsilon) {
            return this.chooseGreedyAction(state);
        }
        // explore
        return this.chooseRandomAction();

    }

    /**
     * given state, get the max action value from q table
     *
     * @param state a State object represent the current state Q(s,a') from Q(s)
     * @return the index of the best action
     */
    int chooseGreedyAction(final State state) {
        double curMax = -Double.MAX_VALUE;
        int curAction = -1;
        final double[] currentActionValue =
                this.qTable.getOrDefault(
                        state,
                        ThreadLocalRandom.current().doubles(this.actionSize, 0, 1).toArray());

        this.qTable.put(state, currentActionValue);

        for (int action = 0; action < currentActionValue.length; action++) {
            if (currentActionValue[action] > curMax) {
                curMax = currentActionValue[action];
                curAction = action;
            }
        }

        return curAction;
    }

    int chooseRandomAction() {
        return new Random().nextInt(this.actionSize);
    }

    void setQValue(final double qValue, final State state, final int action) {
        final double[] value =
                this.qTable.getOrDefault(
                        state,
                        ThreadLocalRandom.current().doubles(this.actionSize, 0, 1).toArray());
        value[action] = qValue;
        this.qTable.put(state, value);
    }

    double getQValue(final State state, final int action) {
        final double[] currentActionValue =
                this.qTable.getOrDefault(
                        state,
                        ThreadLocalRandom.current().doubles(this.actionSize, 0, 1).toArray());
        this.qTable.put(state, currentActionValue);
        return currentActionValue[action];
    }

    /**
     * @param prevDimension an array contains previous states and action info to get previous Q
     *     value Q(s',a')
     * @param curDimension an array contains current states and action info to get current Q value
     *     Q(s,a)
     * @param reward an integer represents reward
     * @param isOnPolicy if true, then use Sarsa. Otherwise, use Q learning
     */
    public void computeQValue(
            final State prevDimension,
            final State curDimension,
            final int prevAction,
            final double reward,
            final boolean isOnPolicy) {

        final double prevQValue = this.getQValue(prevDimension, prevAction);
        final double curQValue = this.getQValue(curDimension, prevAction);

        if (isOnPolicy) {
            // Sarsa
            this.setQValue(
                    prevQValue + learningRate * (reward + discountFactor * curQValue - prevQValue),
                    prevDimension,
                    prevAction);
        } else {
            // Q learning
            final int curAction = this.chooseGreedyAction(curDimension);
            final double maxQValue = this.getQValue(curDimension, curAction);
            this.setQValue(
                    prevQValue + learningRate * (reward + discountFactor * maxQValue - prevQValue),
                    prevDimension,
                    prevAction);
        }
    }

    public int getStateDimension(final State state, final StateName.StateType stateType) {
        return Optional.ofNullable(state.getStateToDimensionMap().get(stateType)).orElse(0);
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
    public void save(final File argFile) {
        PrintStream saveFile = null;
        try {
            saveFile = new PrintStream(new RobocodeFileOutputStream(argFile));
        } catch (final IOException e) {
            System.out.println("Could not create output stream for LUT save file.");
        }

        assert saveFile != null;
        int numEntriesSaved = 0;
        saveFile.println(this.qTable.size());
        for (final Map.Entry<State, double[]> entry : this.qTable.entrySet()) {
            final int[] states = entry.getKey().getIndexedStateValue();
            final double[] actionValues = entry.getValue();
            saveFile.println(states.length);
            for (final int state : states) {
                saveFile.println(state);
            }
            for (int i = 0; i < this.actionSize; ++i) {
                saveFile.println(actionValues[i]);
            }
            numEntriesSaved++;
        }
        saveFile.close();
        System.out.println("Number of LUT table entries saved is " + numEntriesSaved);
    }

    @Override
    public void load(final String argFileName) throws IOException {}

    public void load(final File argFileName) throws IOException {
        if (!argFileName.exists() || argFileName.length() == 0) return;
        final FileInputStream inputFile = new FileInputStream(argFileName);
        final BufferedReader inputReader = new BufferedReader(new InputStreamReader(inputFile));

        int maxIndexFromFile = Integer.parseInt(inputReader.readLine());

        while (maxIndexFromFile-- > 0) {
            final int stateSize = Integer.parseInt(inputReader.readLine());
            final int[] stateValues = new int[stateSize];
            for (int i = 0; i < stateSize; ++i) {
                stateValues[i] = Integer.parseInt(inputReader.readLine());
            }
            final State state = this.getStateFromStateValues(stateValues);
            final double[] actionValues = new double[this.actionSize];
            for (int i = 0; i < this.actionSize; ++i) {
                actionValues[i] = Double.parseDouble(inputReader.readLine());
            }
            this.qTable.put(state, actionValues);
        }
        inputReader.close();
    }

    public State getStateFromStateValues(final int[] indexedStateValue) {
        return ImmutableState.builder()
                .currentHP(StateName.HP.values()[indexedStateValue[0]])
                .currentEnemyHP(StateName.ENEMY_HP.values()[indexedStateValue[1]])
                .currentDistanceToEnemy(StateName.DISTANCE_TO_ENEMY.values()[indexedStateValue[2]])
                .currentDistanceToWall(StateName.DISTANCE_TO_WALL.values()[indexedStateValue[3]])
                .currentEnemyRobotHeading(
                        StateName.ENEMY_ROBOT_HEADING.values()[indexedStateValue[4]])
                .x(indexedStateValue[5])
                .y(indexedStateValue[6])
                .build();
    }
}
