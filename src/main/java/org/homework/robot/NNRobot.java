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
import org.homework.robot.model.State;
import robocode.BattleEndedEvent;
import robocode.RoundEndedEvent;
import robocode.ScannedRobotEvent;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

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
            new NeuralNetArrayImpl(
                    "Default Neural Net",
                    ImmutableState.builder().build(),
                    25,
                    Action.values().length,
                    0.1,
                    0.0,
                    true);
    private static final int MINI_BATCH_SIZE = 20;
    private static final int MEMORY_SIZE = 200;
    private static final String PREV_TRAINED_WEIGHT = "weights.txt";
    private static final ReplayMemory<Memory> replayMemory = new ReplayMemory<>(MEMORY_SIZE);
    private static final List<Double> qTrainRMSE = new ArrayList<>();

    @Override
    public void run() {
        this.setAdjustGunForRobotTurn(true);
        this.setAdjustRadarForGunTurn(true);
        this.load();
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
            neuralNetArray.setQTrainSampleError(0);
            int curTrainSize = 0;
            for (final Object object : replayMemory.randomSample(MINI_BATCH_SIZE)) {
                final Memory experienceBatch = (Memory) object;
                neuralNetArray.qTrain(
                        experienceBatch.getReward(),
                        experienceBatch.getPrevState(),
                        experienceBatch.getPrevAction(),
                        experienceBatch.getCurState());
                curTrainSize++;
            }
            qTrainRMSE.add(
                    neuralNetArray.rmse(neuralNetArray.getQTrainSampleError(), curTrainSize));
        }
    }

    @Override
    public void onRoundEnded(final RoundEndedEvent event) {
        totalRound++;
        this.logWinStatue(100);
        if (totalRound % 100 == 0) {
            log.writeRMSEToFile(this.getDataFile("rmse.log"), qTrainRMSE, 100, true);
            qTrainRMSE.clear();
        }
    }

    /**
     * Get current scanned state
     *
     * @param event The event of scan
     * @return State represent current
     */
    @Override
    public State findCurrentState(final ScannedRobotEvent event) {
        this.setBearing(event.getBearing());
        return ImmutableState.builder()
                .dequantizedState(
                        this.getEnergy() / 100.0,
                        event.getEnergy() / 100.0,
                        event.getDistance() / 1000.0,
                        this.getDistanceFromWall(this.getX(), this.getY()) / 1000.0,
                        event.getBearing() / 360.0,
                        this.getX() / 800.0,
                        this.getY() / 600.0)
                .build();
    }

    @Override
    public void load() {
        try {
            neuralNetArray.load(this.getDataFile(PREV_TRAINED_WEIGHT));
            System.out.println("Loaded");
        } catch (final IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void onBattleEnded(final BattleEndedEvent event) {
        neuralNetArray.save(this.getDataFile((PREV_TRAINED_WEIGHT)), true);
    }
}
