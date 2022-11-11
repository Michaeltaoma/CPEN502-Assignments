package org.homework;

import org.homework.robot.model.ImmutableState;
import org.homework.robot.model.State;
import org.homework.robot.model.StateName;

public class Main {
    public static void main(final String[] args) {
        final State state = ImmutableState.builder().currentHP(StateName.HP.HIGH).build();
    }
}
