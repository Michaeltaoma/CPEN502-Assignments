package org.homework.robot.model;

import org.immutables.value.Value;

@Value.Immutable
public abstract class Memory {
    @Value.Default
    public State getPrevState() {
        return ImmutableState.builder().build();
    }

    @Value.Default
    public State getCurState() {
        return ImmutableState.builder().build();
    }

    @Value.Default
    public double getReward() {
        return 1.0;
    }

    @Value.Default
    public Action getPrevAction() {
        return Action.values()[0];
    }
}
