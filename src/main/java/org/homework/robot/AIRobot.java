package org.homework.robot;

import robocode.AdvancedRobot;
import robocode.BattleEndedEvent;
import robocode.BulletHitEvent;
import robocode.BulletMissedEvent;
import robocode.DeathEvent;
import robocode.HitByBulletEvent;
import robocode.HitRobotEvent;
import robocode.HitWallEvent;
import robocode.RobocodeFileOutputStream;
import robocode.ScannedRobotEvent;
import robocode.WinEvent;

import java.awt.*;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Date;

import static org.homework.Util.closeOutputStream;

public class AIRobot extends AdvancedRobot {
    private RobocodeFileOutputStream robocodeFileOutputStream = null;
    private boolean colorChangeSwitch = true;

    @Override
    public void run() {
        this.initRobocodeFileOutputStream();

        while (true) {
            this.scan();
            this.act();
        }
    }

    /**
     * switch mode (scan or move) when move: choose action -> move -> update Q when scan: turn
     * Currently it just do dumb thing
     */
    private void act() {
        this.setBodyColor(this.colorChangeSwitch ? Color.pink : Color.black);
        this.colorChangeSwitch = !this.colorChangeSwitch;
    }

    @Override
    public void onScannedRobot(final ScannedRobotEvent event) {
        this.info(
                String.format("Event: %s, Energy bearing: %f", event.getName(), event.getEnergy()));
    }

    public void calculateQValue(final boolean isOffPolicy) {
        // calculate q
        // update q
    }

    @Override
    public void onHitByBullet(final HitByBulletEvent e) {
        // update reward
    }

    @Override
    public void onBulletHit(final BulletHitEvent e) {
        // update reward
    }

    @Override
    public void onBulletMissed(final BulletMissedEvent e) {
        // update reward
    }

    @Override
    public void onHitWall(final HitWallEvent e) {
        // update reward
    }

    @Override
    public void onHitRobot(final HitRobotEvent e) {
        // update reward
    }

    @Override
    public void onWin(final WinEvent e) {
        // update reward
    }

    @Override
    public void onDeath(final DeathEvent e) {
        // update reward
    }

    @Override
    public void onBattleEnded(final BattleEndedEvent event) {
        closeOutputStream(this.robocodeFileOutputStream);
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

        this.info("Successfully initialized robocode logger");
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
