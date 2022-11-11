package org.homework.robot;

import org.homework.robot.model.Action;
import org.homework.robot.model.ImmutableState;
import org.homework.robot.model.State;
import org.homework.robot.model.StateName;
import robocode.AdvancedRobot;
import robocode.BattleEndedEvent;
import robocode.RobocodeFileOutputStream;
import robocode.ScannedRobotEvent;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Date;

import static org.homework.Util.closeOutputStream;

public class AIRobot extends AdvancedRobot {
    private RobocodeFileOutputStream robocodeFileOutputStream = null;

    private boolean doesScanRun = false;
    private Action currentAction = Action.AHEAD;

    @Override
    public void run() {
        this.initRobocodeFileOutputStream();

        while (true) {
            this.setTurnRadarLeftRadians(2 * Math.PI);
            this.scan();
            this.act();
        }
    }

    /**
     * switch mode (scan or move) when move: choose action -> move -> update Q when scan: turn
     * Currently it just do dumb thing
     */
    private void act() {
        this.setTurnLeft(this.currentAction.getDirection()[0]);
        this.setTurnRight(this.currentAction.getDirection()[1]);
        this.setAhead(this.currentAction.getDirection()[2]);
        this.execute();
    }

    @Override
    public void onScannedRobot(final ScannedRobotEvent event) {
        final State currentState = this.getCurrentState(event);
        this.doesScanRun = !this.doesScanRun;
        this.info(String.format("Current State: %s\n", currentState));
        this.currentAction = this.chooseCurrentAction();
    }

    public State getCurrentState(final ScannedRobotEvent event) {

        return ImmutableState.builder()
                .currentHP(this.toHP(this.getEnergy()))
                .distanceToEnemy(this.toDistanceToEnemy(event.getDistance()))
                .build();
    }

    public Action chooseCurrentAction() {
        return this.doesScanRun ? Action.AHEAD : Action.TURN_LEFT;
    }

    @Override
    public void onBattleEnded(final BattleEndedEvent event) {
        closeOutputStream(this.robocodeFileOutputStream);
    }

    StateName.HP toHP(final double hp) {
        if (hp < 30) {
            return StateName.HP.LOW;
        } else if (hp < 60) {
            return StateName.HP.MID;
        } else {
            return StateName.HP.HIGH;
        }
    }

    StateName.DISTANCE_TO_ENEMY toDistanceToEnemy(final double distance) {
        if (distance < 30) {
            return StateName.DISTANCE_TO_ENEMY.LOW;
        } else if (distance < 60) {
            return StateName.DISTANCE_TO_ENEMY.MID;
        } else {
            return StateName.DISTANCE_TO_ENEMY.HIGH;
        }
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
