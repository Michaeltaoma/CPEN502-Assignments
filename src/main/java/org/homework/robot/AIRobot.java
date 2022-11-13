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
import robocode.HitRobotEvent;
import robocode.RobocodeFileOutputStream;
import robocode.ScannedRobotEvent;
import robocode.WinEvent;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collections;
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
    private double bearing = 0.0;

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
        if (this.currentAction == Action.FIRE) {
            this.setTurnGunRight(this.getHeading() - this.getGunHeading() + this.bearing);
            this.setFire(this.currentAction.getDirection()[3]);
        }
        this.execute();
    }

    /**
     * Get current scanned state
     *
     * @param event The event of scan
     * @return State represent current
     */
    public State getCurrentState(final ScannedRobotEvent event) {
        this.bearing = event.getBearing();

        return ImmutableState.builder()
                .currentHP(StateName.HP.values()[this.toCategoricalState(event.getEnergy())])
                .currentEnemyHP(
                        StateName.ENEMY_HP.values()[this.toCategoricalState(event.getBearing())])
                .currentDistanceToEnemy(
                        StateName.DISTANCE_TO_ENEMY
                                .values()[this.toCategoricalState(event.getDistance())])
                .currentDistanceToWall(
                        StateName.DISTANCE_TO_WALL
                                .values()[
                                this.toCategoricalState(
                                        this.getDistanceFromWall(this.getX(), this.getY()))])
                .build();
    }

    /**
     * Calculate distance from wall
     *
     * @param x1 x1
     * @param y1
     * @return
     */
    public double getDistanceFromWall(final double x1, final double y1) {
        final double width = this.getBattleFieldWidth();
        final double height = this.getBattleFieldHeight();
        final double disb = height - y1, disl = x1, disr = width - x1;

        return Collections.max(Arrays.asList(y1, disb, disl, disr));
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
    public void onHitRobot(final HitRobotEvent event) {
        this.reward -= 4 * BASIC_REWARD;
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
