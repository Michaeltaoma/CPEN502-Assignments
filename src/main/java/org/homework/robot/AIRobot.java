package org.homework.robot;

import lombok.Setter;
import org.homework.rl.LUTImpl;
import org.homework.robot.model.Action;
import org.homework.robot.model.ImmutableState;
import org.homework.robot.model.State;
import org.homework.robot.model.StateName;
import org.homework.util.LogFile;
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

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;

@Setter
public class AIRobot extends AdvancedRobot {
    private static final double BASIC_REWARD = .5;
    public static LogFile log = new LogFile();
    private static int winRound = 0;
    private static int totalRound = 0;
    private static int rounds = 0;
    private final boolean isOnPolicy = false;
    private boolean isImmediateReward = true;
    private double reward = .0;
    private State currentState = ImmutableState.builder().build();
    private State prevState = ImmutableState.builder().from(this.currentState).build();
    private LUTImpl lut = new LUTImpl(this.currentState);
    private Action currentAction;
    private double bearing = 0.0;
    private RobocodeFileOutputStream robocodeFileOutputStream;

    @Override
    public void run() {
        this.load();
        this.setAdjustGunForRobotTurn(true);
        this.setAdjustRadarForGunTurn(true);
        while (true) {
            this.setTurnRadarLeftRadians(2 * Math.PI);
            this.scan();
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
        if (this.currentAction.name().contains("FIRE")) {
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
                .currentHP(StateName.HP.values()[this.toCategoricalState(this.getEnergy(), 30, 2)])
                .currentDistanceToWall(
                        StateName.DISTANCE_TO_WALL
                                .values()[
                                this.toCategoricalState(
                                        this.getDistanceFromWall(this.getX(), this.getY()), 30, 2)])
                .x(StateName.X.values()[this.toCategoricalState(this.getX(), 265, 2)])
                .y(StateName.Y.values()[this.toCategoricalState(this.getY(), 200, 2)])
                .currentEnemyHP(
                        StateName.ENEMY_HP
                                .values()[this.toCategoricalState(event.getBearing(), 30, 2)])
                .currentDistanceToEnemy(
                        StateName.DISTANCE_TO_ENEMY
                                .values()[this.toCategoricalState(event.getDistance(), 30, 2)])
                .currentEnemyRobotHeading(
                        StateName.ENEMY_ROBOT_HEADING
                                .values()[this.toCategoricalState(event.getHeading(), 120, 2)])
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
        this.prevState = ImmutableState.builder().from(this.currentState).build();
        this.currentState = ImmutableState.builder().from(this.getCurrentState(event)).build();
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
    public void onBattleEnded(final BattleEndedEvent event) {}

    @Override
    public void onRoundEnded(final RoundEndedEvent event) {
        this.lut.save(this.getDataFile(this.getEntryFileName()));
        totalRound++;
        this.logWinStatue(100);
    }

    void logWinStatue(final int round) {
        if ((totalRound % round == 0) && (totalRound != 0)) {
            final double winPercentage = (double) winRound / round;
            final File folderDst1 = this.getDataFile(this.getWinRoundLogFileName());
            log.writeToFile(folderDst1, winPercentage, ++rounds);
            winRound = 0;
        }
    }

    private String getWinRoundLogFileName() {
        return String.format(
                "win-round-%s-%s",
                this.isOnPolicy ? "OnPolicy" : "OffPolicy",
                this.isImmediateReward ? "ImmediateReward" : "TerminalReward");
    }

    /**
     * Call when battle win
     *
     * @param event
     */
    @Override
    public void onWin(final WinEvent event) {
        this.reward += this.isImmediateReward ? 10 * BASIC_REWARD : 20 * BASIC_REWARD;
        this.updateQValue();
        winRound++;
    }

    /**
     * Call when robot die
     *
     * @param event
     */
    @Override
    public void onHitRobot(final HitRobotEvent event) {
        this.reward -= this.isImmediateReward ? 2 * BASIC_REWARD : 0;
    }

    /**
     * Call when robot hits wall
     *
     * @param event
     */
    @Override
    public void onHitWall(final HitWallEvent event) {
        this.reward -= this.isImmediateReward ? 2 * BASIC_REWARD : 0;
    }

    /**
     * This method is called when one of your bullets hits another main.robot
     *
     * @param event
     */
    @Override
    public void onBulletHit(final BulletHitEvent event) {
        this.reward += this.isImmediateReward ? 2 * BASIC_REWARD : 0;
    }

    /**
     * This method is called when one of our bullets has missed.
     *
     * @param event
     */
    @Override
    public void onBulletMissed(final BulletMissedEvent event) {
        this.reward -= this.isImmediateReward ? 1 * BASIC_REWARD : 0;
    }

    /**
     * This method is called when your main.robot is hit by a bullet.
     *
     * @param event
     */
    @Override
    public void onHitByBullet(final HitByBulletEvent event) {
        this.reward -= this.isImmediateReward ? 2 * BASIC_REWARD : 0;
    }

    /**
     * Call when robot die
     *
     * @param event
     */
    @Override
    public void onDeath(final DeathEvent event) {
        this.reward -= this.isImmediateReward ? 10 * BASIC_REWARD : 20 * BASIC_REWARD;
        this.updateQValue();
    }

    /**
     * Helper function to map numerical value to categorical value
     *
     * @param val numerical state value
     * @return ordinal for the state
     */
    public int toCategoricalState(final double val, final int splitter, final int maxIndex) {
        return Math.min(maxIndex, new Double(val).intValue() / splitter);
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

    /** Load previous saved log file */
    private void load() {
        try {
            this.lut.load(this.getDataFile(this.getEntryFileName()));
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
        return String.format("AIRobot-%s-robot.txt", "crazy");
    }
}
