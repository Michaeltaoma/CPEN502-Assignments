package org.homework.robot.model;

public class StateName {
    public enum StateType {
        MY_HP(3),
        ENEMY_HP(3),
        DISTANCE_TO_ENEMY(3),
        DISTANCE_TO_WALL(3),
        ENEMY_ROBOT_HEADING(3);

        final int numTypes;

        StateType(final int _numTypes) {
            this.numTypes = _numTypes;
        }

        public int getNumTypes() {
            return this.numTypes;
        }
    }

    public enum HP {
        LOW,
        MID,
        HIGH
    }

    public enum ENEMY_HP {
        LOW,
        MID,
        HIGH
    }

    public enum DISTANCE_TO_ENEMY {
        LOW,
        MID,
        HIGH
    }

    public enum DISTANCE_TO_WALL {
        LOW,
        MID,
        HIGH
    }

    public enum ENEMY_ROBOT_HEADING {
        LOW,
        MID,
        HIGH
    }
}
