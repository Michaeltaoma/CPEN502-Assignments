package org.homework.robot;

import org.homework.rl.LUTImpl;
import robocode.*;
import robocode.Robot;

import java.awt.*;
import java.util.Arrays;

public class AIRobot extends Robot {

  public static void main(final String[] args) {

    final double[] currentState = new double[] {1.0, 2.0, 100.0, 100.0};
    final double[] nextState = Action.Fire.toNextPos(currentState);

    System.out.printf(
        "Previous: %s\n Current: %s", Arrays.toString(currentState), Arrays.toString(nextState));

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
       * switch mode (scan or move) when move: choose action -> move -> update Q when scan: turn
       * radar??
       */
    }
  }

  public void act(Action action) {
    switch (action) {
      case Fire:
        {
        }
        break;
    }
  }

  public void calculateQValue(boolean isOffPolicy) {
    // calculate q
    // update q
  }

  public void onScannedRobot(final ScannedRobotEvent e) {
    final double enemyBearing = e.getBearing();
  }

  @Override
  public void onHitByBullet(HitByBulletEvent e) {
    // update reward
  }

  @Override
  public void onBulletHit(BulletHitEvent e) {
    // update reward
  }

  @Override
  public void onBulletMissed(BulletMissedEvent e) {
    // update reward
  }

  @Override
  public void onHitWall(HitWallEvent e) {
    // update reward
  }

  @Override
  public void onHitRobot(HitRobotEvent e) {
    // update reward
  }

  @Override
  public void onWin(WinEvent e) {
    // update reward
  }

  @Override
  public void onDeath(DeathEvent e) {
    // update reward
  }
}