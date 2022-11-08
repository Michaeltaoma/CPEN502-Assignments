package org.homework.robot;

public enum Action {
    West(new double[] {-100.0, 0.0, 0.0, 0.0}),
    East(new double[] {100.0, 0.0, 0.0, 0.0}),
    North(new double[] {0.0, 100.0, 0.0, 0.0}),
    South(new double[] {0.0, -100.0, 0.0, 0.0}),
    Fire(new double[] {0.0, 0.0, 0.0, -100.0});

    final double[] direction;

    Action(final double[] direction) {
        this.direction = direction;
    }

    double[] getDirection() {
        return this.direction;
    }

    double[] toNextPos(final double... currentState) {
        final double[] currentStateCopy = currentState.clone();
        for (int index = 0; index < currentState.length; index++) {
            currentStateCopy[index] += this.getDirection()[index];
        }
        return currentStateCopy;
    }
}
