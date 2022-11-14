package org.homework.robot;

import lombok.Setter;
import org.homework.rl.LUTImpl;
import org.homework.robot.model.Action;
import org.homework.robot.model.ImmutableState;
import org.homework.robot.model.State;
import org.homework.robot.model.StateName;
import robocode.AdvancedRobot;
import robocode.BattleEndedEvent;
import robocode.BulletHitEvent;
import robocode.BulletMissedEvent;
import robocode.DeathEvent;
import robocode.HitByBulletEvent;
import robocode.HitRobotEvent;
import robocode.HitWallEvent;
import robocode.RobocodeFileOutputStream;
import robocode.RoundEndedEvent;
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
    private int winRound = 0;
    private int totalRound = 0;

    @Override
    public void run() {
        this.initRobocodeFileOutputStream();
        this.load();
        while (true) {
            this.setTurnRadarLeftRadians(2 * Math.PI);
            this.currentAction = this.chooseCurrentAction();
            this.act();
            this.updateQValue();
            this.reward = .0;
        }
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
     * Get Current State when the enemy is not scanned
     *
     * @return Current State
     */
    public State getCurrentState() {
        return ImmutableState.builder()
                .from(this.currentState)
                .currentHP(StateName.HP.values()[this.toCategoricalState(this.getEnergy())])
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

    /** Update q value */
    public void updateQValue() {
        this.lut.computeQValue(
                this.prevState,
                this.currentState,
                this.currentAction.ordinal(),
                this.reward,
                this.isOnPolicy);
    }

    /**
     * Update robot's current state and previous state when the enemy is scanned
     *
     * @param event event when a enemy robot is being scanned
     */
    private void updateRobotState(final ScannedRobotEvent event) {
        this.prevState = this.currentState;
        this.currentState = this.getCurrentState(event);
    }

    /** Update robot's current state and previous state */
    private void updateRobotState() {
        this.prevState = this.currentState;
        this.currentState = this.getCurrentState();
    }

    private void updateRound(final boolean isWin) {
        if (isWin) {
            this.winRound++;
        }
        this.totalRound++;
    }

    /** Update current robot state and update q value */
    private void updateLearning() {
        this.updateRobotState();
        this.updateQValue();
    }

    /** Called when the enemy robot has been scanned */
    @Override
    public void onScannedRobot(final ScannedRobotEvent event) {
        this.updateRobotState(event);
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

    @Override
    public void onRoundEnded(final RoundEndedEvent event) {
        this.info(String.format("Entry: %d on end", this.lut.qTable.size()));
        this.lut.save(this.getDataFile(this.getEntryFileName()));
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
        this.updateLearning();
        this.updateRound(true);
    }

    /**
     * Call when robot die
     *
     * @param event
     */
    @Override
    public void onHitRobot(final HitRobotEvent event) {
        this.reward -= BASIC_REWARD;
        this.updateLearning();
    }

    /**
     * Call when robot hits wall
     *
     * @param event
     */
    @Override
    public void onHitWall(final HitWallEvent event) {
        this.reward -= BASIC_REWARD;
        this.updateLearning();
    }

    /**
     * This method is called when one of your bullets hits another main.robot
     *
     * @param event
     */
    @Override
    public void onBulletHit(final BulletHitEvent event) {
        this.reward += 2 * BASIC_REWARD;
        this.updateLearning();
    }

    /**
     * This method is called when one of our bullets has missed.
     *
     * @param event
     */
    @Override
    public void onBulletMissed(final BulletMissedEvent event) {
        this.reward -= 2 * BASIC_REWARD;
        this.updateLearning();
    }

    /**
     * This method is called when your main.robot is hit by a bullet.
     *
     * @param event
     */
    @Override
    public void onHitByBullet(final HitByBulletEvent event) {
        this.reward -= 1 * BASIC_REWARD;
        this.updateLearning();
    }

    /**
     * Call when robot die
     *
     * @param event
     */
    @Override
    public void onDeath(final DeathEvent event) {
        this.reward -= 10 * BASIC_REWARD;
        this.updateLearning();
        this.updateRound(false);
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

    private void load() {
        try {
            this.lut.load(this.getDataFile(this.getEntryFileName()));
            this.info(String.format("qtable contains %d entries", this.lut.qTable.size()));
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

    /**
     * Get file that store the weight
     *
     * @return filename that store the weight
     */
    private String getEntryFileName() {
        return "AIRobot-crazy-robot.txt";
    }
}
