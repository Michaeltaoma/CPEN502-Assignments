/*
 * Copyright (c) 2001-2022 Mathew A. Nelson and Robocode contributors
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://robocode.sourceforge.io/license/epl-v10.html
 */
package org.homework.robot;

import org.homework.neuralnet.NeuralNetArrayImpl;
import org.homework.replaymemory.ReplayMemory;
import org.homework.robot.model.Action;
import org.homework.robot.model.ImmutableMemory;
import org.homework.robot.model.ImmutableState;
import org.homework.robot.model.Memory;

/**
 * Corners - a sample robot by Mathew Nelson.
 *
 * <p>This robot moves to a corner, then swings the gun back and forth. If it dies, it tries a new
 * corner in the next round.
 *
 * @author Mathew A. Nelson (original)
 * @author Flemming N. Larsen (contributor)
 */
public class NNRobot extends AIRobot {
    private static final NeuralNetArrayImpl neuralNetArray =
            new NeuralNetArrayImpl(ImmutableState.builder().build());
    private static final int MINI_BATCH_SIZE = 20;
    private static final int MEMORY_SIZE = 100;

    private static final String PREV_TRAINED_WEIGHT = "";
    private static final ReplayMemory<Memory> replayMemory = new ReplayMemory<>(MEMORY_SIZE);

    @Override
    public void run() {
        this.setAdjustGunForRobotTurn(true);
        this.setAdjustRadarForGunTurn(true);
        neuralNetArray.load(this.getDataFile(PREV_TRAINED_WEIGHT));
        while (true) {
            this.setTurnRadarLeftRadians(2 * Math.PI);
            this.scan();
            this.setCurrentAction(this.chooseCurrentAction());
            this.act();
            this.updateQValue();
            this.setReward(.0);
        }
    }

    @Override
    public Action chooseCurrentAction() {
        return Action.values()[neuralNetArray.chooseAction(currentState)];
    }

    @Override
    /** Update q value */
    public void updateQValue() {
        final Memory memory =
                ImmutableMemory.builder()
                        .curState(currentState)
                        .prevState(prevState)
                        .reward(this.getReward())
                        .prevAction(this.getCurrentAction())
                        .build();
        replayMemory.add(memory);
        if (replayMemory.sizeOf() >= MINI_BATCH_SIZE) {
            for (final Object object : replayMemory.randomSample(MINI_BATCH_SIZE)) {
                final Memory experienceBatch = (Memory) object;
                neuralNetArray.qTrain(
                        experienceBatch.getReward(),
                        experienceBatch.getPrevState(),
                        experienceBatch.getPrevAction(),
                        experienceBatch.getCurState());
            }
        }
    }
}
