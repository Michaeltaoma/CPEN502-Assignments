package org.homework.robot.model;

public enum Action {
    TO_LEFT(new double[] {50.0, 0.0, 50.0, 0.0}),
    TO_RIGHT(new double[] {0.0, 50.0, 50.0, 0.0}),
    AHEAD(new double[] {0.0, 0.0, 50.0, 0.0}),
    BACK(new double[] {0.0, 0.0, -50.0, 0.0}),
    FIRE(new double[] {50.0, 0.0, 50.0, 3.0});

    final double[] direction;

    Action(final double[] direction) {
        this.direction = direction;
    }

    public double[] getDirection() {
        return this.direction;
    }
}
