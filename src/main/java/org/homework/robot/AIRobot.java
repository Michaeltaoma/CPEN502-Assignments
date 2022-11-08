package org.homework.robot;

import robocode.BulletHitEvent;
import robocode.BulletMissedEvent;
import robocode.DeathEvent;
import robocode.HitByBulletEvent;
import robocode.HitRobotEvent;
import robocode.HitWallEvent;
import robocode.Robot;
import robocode.ScannedRobotEvent;
import robocode.WinEvent;

import java.awt.*;
import java.util.Arrays;

public class AIRobot extends Robot {

    public static void main(final String[] args) {

        final double[] currentState = new double[] {1.0, 2.0, 100.0, 100.0};
        final double[] nextState = Action.Fire.toNextPos(currentState);

        System.out.printf(
                "Previous: %s\n Current: %s",
                Arrays.toString(currentState), Arrays.toString(nextState));

        final double alpha = 0.3; // learning rate
        final double gamma = 0.9; // discount factor
        final boolean isOffPolicy = true; // on/off policy

        //    public LUTImpl qTable = new LUTImpl();
    }

    public void run() {
        this.setAdjustGunForRobotTurn(true);
        this.setBodyColor(Color.pink);
        this.setRadarColor(Color.orange);

        while (true) {
            /**
             * switch mode (scan or move) when move: choose action -> move -> update Q when scan:
             * turn radar??
             */
        }
    }

    public void act(final Action action) {
        if (action == Action.Fire) {}
    }

    public void calculateQValue(final boolean isOffPolicy) {
        // calculate q
        // update q
    }

    public void onScannedRobot(final ScannedRobotEvent e) {
        final double enemyBearing = e.getBearing();
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
}
