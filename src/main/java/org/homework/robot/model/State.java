package org.homework.robot.model;

import com.google.common.collect.ImmutableMap;
import org.immutables.value.Value;

import java.util.Arrays;

import static org.homework.robot.model.StateName.StateType.DISTANCE_TO_ENEMY;
import static org.homework.robot.model.StateName.StateType.DISTANCE_TO_WALL;
import static org.homework.robot.model.StateName.StateType.ENEMY_HP;
import static org.homework.robot.model.StateName.StateType.ENEMY_ROBOT_HEADING;
import static org.homework.robot.model.StateName.StateType.MY_HP;

@Value.Immutable
public abstract class State {
    @Value.Default
    public StateName.HP getCurrentHP() {
        return StateName.HP.MID;
    }

    @Value.Default
    public StateName.ENEMY_HP getCurrentEnemyHP() {
        return StateName.ENEMY_HP.MID;
    }

    @Value.Default
    public StateName.DISTANCE_TO_ENEMY getCurrentDistanceToEnemy() {
        return StateName.DISTANCE_TO_ENEMY.MID;
    }

    @Value.Default
    public StateName.DISTANCE_TO_WALL getCurrentDistanceToWall() {
        return StateName.DISTANCE_TO_WALL.MID;
    }

    @Value.Default
    public StateName.ENEMY_ROBOT_HEADING getCurrentEnemyRobotHeading() {
        return StateName.ENEMY_ROBOT_HEADING.MID;
    }

    @Value.Default
    public int getX() {
        return 50;
    }

    @Value.Default
    public int getY() {
        return 50;
    }

    @Value.Default
    public int[] getIndexedStateValue() {
        return new int[] {
            this.getCurrentHP().ordinal(),
            this.getCurrentEnemyHP().ordinal(),
            this.getCurrentDistanceToEnemy().ordinal(),
            this.getCurrentDistanceToWall().ordinal(),
            this.getCurrentEnemyRobotHeading().ordinal(),
            this.getX(),
            this.getY()
        };
    }

    @Value.Default
    public ImmutableMap<StateName.StateType, Integer> getStateToDimensionMap() {
        return ImmutableMap.<StateName.StateType, Integer>builder()
                .put(MY_HP, MY_HP.getNumTypes())
                .put(ENEMY_HP, ENEMY_HP.getNumTypes())
                .put(DISTANCE_TO_ENEMY, DISTANCE_TO_ENEMY.getNumTypes())
                .put(DISTANCE_TO_WALL, DISTANCE_TO_WALL.getNumTypes())
                .put(ENEMY_ROBOT_HEADING, ENEMY_ROBOT_HEADING.getNumTypes())
                .build();
    }

    @Override
    public String toString() {
        return String.format(
                "Current State Array: %s", Arrays.toString(this.getIndexedStateValue()));
    }

    @Override
    public boolean equals(final Object obj) {
        if (obj == this) return true;

        if (!(obj instanceof State)) return false;

        final State state = (State) obj;

        return Arrays.equals(this.getIndexedStateValue(), state.getIndexedStateValue());
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(this.getIndexedStateValue());
    }
}
