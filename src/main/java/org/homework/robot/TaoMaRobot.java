package org.homework.robot;

import robocode.Robot;
import robocode.ScannedRobotEvent;

import java.awt.*;

public class TaoMaRobot extends Robot {
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
