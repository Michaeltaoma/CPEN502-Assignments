package org.homework.robot;

import lombok.Setter;
import org.homework.rl.LUTImpl;
import org.homework.robot.model.Action;
import org.homework.robot.model.ImmutableState;
import org.homework.robot.model.State;
import org.homework.robot.model.StateName;
import robocode.AdvancedRobot;
import robocode.BattleEndedEvent;
import robocode.DeathEvent;
import robocode.RobocodeFileOutputStream;
import robocode.ScannedRobotEvent;
import robocode.WinEvent;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Date;

import static org.homework.Util.closeOutputStream;

@Setter
public class AIRobot extends AdvancedRobot {
    private static final double BASIC_REWARD = .5;
    private final boolean isOnPolicy = false;
    private double reward = .0;
    private RobocodeFileOutputStream robocodeFileOutputStream = null;
    private State currentState = ImmutableState.builder().build();
    private State prevState = this.currentState;
    private LUTImpl lut = new LUTImpl(this.currentState);
    private Action currentAction;

    @Override
    public void run() {
        this.initRobocodeFileOutputStream();
        while (true) {
            this.setTurnRadarLeftRadians(2 * Math.PI);
            this.scan();
            this.currentAction = this.chooseCurrentAction();
            this.act();
            this.updateQValue();
            this.reward = .0;
        }
    }

    /** Update q value */
    public void updateQValue() {
        this.lut.computeQValue(
                this.prevState,
                this.currentState,
                this.currentAction.ordinal(),
                this.reward,
                this.isOnPolicy);
    }

    /** Act based on action */
    private void act() {
        this.setTurnLeft(this.currentAction.getDirection()[0]);
        this.setTurnRight(this.currentAction.getDirection()[1]);
        this.setAhead(this.currentAction.getDirection()[2]);
        this.info(String.format("Current Action: %s", this.currentAction.name()));
        this.execute();
    }

    /**
     * Get current scanned state
     *
     * @param event The event of scan
     * @return State represent current
     */
    public State getCurrentState(final ScannedRobotEvent event) {
        return ImmutableState.builder()
                .currentHP(StateName.HP.values()[this.toCategoricalState(event.getEnergy())])
                .currentEnemyHP(
                        StateName.ENEMY_HP.values()[this.toCategoricalState(event.getBearing())])
                .currentDistanceToEnemy(
                        StateName.DISTANCE_TO_ENEMY
                                .values()[this.toCategoricalState(event.getDistance())])
                .currentDistanceToWall(
                        StateName.DISTANCE_TO_WALL
                                .values()[this.toCategoricalState(this.getDistanceRemaining())])
                .build();
    }

    /**
     * Choose action based on current state
     *
     * @return Action the robot should do
     */
    public Action chooseCurrentAction() {
        return Action.values()[this.lut.chooseAction(this.currentState)];
    }

    /**
     * Called when the enemy robot has been scanned
     *
     * @param event
     */
    @Override
    public void onScannedRobot(final ScannedRobotEvent event) {
        this.prevState = this.currentState;
        this.currentState = this.getCurrentState(event);
        this.info(String.format("Current State: %s\n", this.currentState));
    }

    /**
     * Called when battle end
     *
     * @param event
     */
    @Override
    public void onBattleEnded(final BattleEndedEvent event) {
        closeOutputStream(this.robocodeFileOutputStream);
    }

    /**
     * Call when battle win
     *
     * @param event
     */
    @Override
    public void onWin(final WinEvent event) {
        this.reward += 10 * BASIC_REWARD;
        this.updateQValue();
    }

    /**
     * Call when robot die
     *
     * @param event
     */
    @Override
    public void onDeath(final DeathEvent event) {
        this.reward -= 10 * BASIC_REWARD;
        this.updateQValue();
    }

    /**
     * Helper function to map numerical value to categorical value
     *
     * @param val numerical state value
     * @return ordinal for the state
     */
    public int toCategoricalState(final double val) {
        return Math.min(2, new Double(val).intValue() / 30);
    }

    /**
     * Log string to the log file created, easier for debug
     *
     * @param msg message to be logged
     */
    private void info(final String msg) {
        try {
            this.robocodeFileOutputStream.write(
                    String.format("%s\n", msg).getBytes(StandardCharsets.UTF_8));
        } catch (final IOException e) {
            throw new RuntimeException(e);
        }
    }

    /** initialize the logger for the robot */
    private void initRobocodeFileOutputStream() {
        if (this.robocodeFileOutputStream != null) return;

        final String targetLogFilePath = this.getLogFileName();

        try {
            this.robocodeFileOutputStream =
                    new RobocodeFileOutputStream(this.getDataFile(targetLogFilePath));
        } catch (final IOException e) {
            throw new RuntimeException(e);
        }

        this.info(
                String.format("Successfully initialized robocode logger: %s\n", targetLogFilePath));
    }

    /**
     * Get log file name. Naming convention: robot + current time in ms The log file will be created
     * by robocode in robocode/robots/.data/org/homework/robot/AIRobot.data
     *
     * @return name
     */
    private String getLogFileName() {
        return String.format("robot-%d.txt", new Date().getTime());
    }
}
