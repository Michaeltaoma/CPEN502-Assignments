package org.homework.robot.model;

import org.immutables.value.Value;

@Value.Immutable
public abstract class State {
    @Value.Default
    StateName.HP getCurrentHP() {
        return StateName.HP.MID;
    }

    @Value.Default
    StateName.DISTANCE_TO_ENEMY getDistanceToEnemy() {
        return StateName.DISTANCE_TO_ENEMY.MID;
    }

    @Value.Default
    int[] getDimensionArray() {
        return new int[] {
            StateName.HP.values().length, StateName.DISTANCE_TO_ENEMY.values().length
        };
    }

    @Value.Default
    int[] getIndexedStateValue() {
        return new int[] {this.getCurrentHP().ordinal(), this.getDistanceToEnemy().ordinal()};
    }

    @Override
    public String toString() {
        return String.format(
                "HP: %s, Distance to enemy: %s",
                this.getCurrentHP().name(), this.getDistanceToEnemy().name());
    }
}
