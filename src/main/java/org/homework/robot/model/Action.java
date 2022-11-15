package org.homework.robot.model;

public enum Action {
    AHEAD_LEFT(new double[] {50.0, 0.0, 50.0, 0.0}),
    AHEAD_RIGHT(new double[] {0.0, 50.0, 50.0, 0.0}),
    BACK_LEFT(new double[] {50.0, 0.0, -50.0, 0.0}),
    BACK_RIGHT(new double[] {0.0, 50.0, -50.0, 0.0}),
    AHEAD(new double[] {0.0, 0.0, 50.0, 0.0}),
    BACK(new double[] {0.0, 0.0, -50.0, 0.0}),
    HEAVY_FIRE(new double[] {0.0, 0.0, 0.0, 3.0}),
    LIGHT_FIRE(new double[] {0.0, 0.0, 0.0, 1.0});

    final double[] direction;

    Action(final double[] direction) {
        this.direction = direction;
    }

    public double[] getDirection() {
        return this.direction;
    }
}
