package org.homework.robot;

import robocode.Robot;
import robocode.ScannedRobotEvent;

import java.awt.*;
import java.util.Arrays;

public class AIRobot extends Robot {

  public static void main(final String[] args) {
    //
    final double[] currentState = new double[] {1.0, 2.0, 100.0, 100.0};

    final double[] nextState = Action.Fire.toNextPos(currentState);

    System.out.printf(
        "Previous: %s\n Current: %s", Arrays.toString(currentState), Arrays.toString(nextState));
  }

  public void run() {
    this.setAdjustGunForRobotTurn(true);
    this.setBodyColor(Color.pink);
    this.setRadarColor(Color.orange);

    while (true) {
      this.turnRadarRight(90);
      this.ahead(100);
      this.turnRight(90);
    }
  }

  public void onScannedRobot(final ScannedRobotEvent e) {
    final double enemyBearing = e.getBearing();
  }
}
