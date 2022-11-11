package org.homework.robot.model;

public enum Action {
    TURN_LEFT(new double[] {90.0, 0.0, 0.0}),
    TURN_RIGHT(new double[] {0.0, 90.0, 0.0}),
    AHEAD(new double[] {0.0, 0.0, 100.0});

    final double[] direction;

    Action(final double[] direction) {
        this.direction = direction;
    }

    public double[] getDirection() {
        return this.direction;
    }
}
