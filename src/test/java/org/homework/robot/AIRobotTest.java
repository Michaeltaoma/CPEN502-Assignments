package org.homework.robot;

import org.junit.jupiter.api.BeforeEach;

class AIRobotTest {

    AIRobot testAIRobot;

    @BeforeEach
    void setUp() {
        this.testAIRobot = new AIRobot();
    }

    //    @Test
    //    void TEST_CURRENT_STATE_MAPPER() {
    //        final ScannedRobotEvent mockScannedRobotEvent = mock(ScannedRobotEvent.class);
    //
    //        when(mockScannedRobotEvent.getEnergy()).thenReturn(10.0);
    //        when(mockScannedRobotEvent.getDistance()).thenReturn(50.0);
    //
    //        assertEquals(
    //                this.testAIRobot.getCurrentState(mockScannedRobotEvent).getCurrentHP(),
    //                StateName.HP.LOW);
    //        assertEquals(
    //
    // this.testAIRobot.getCurrentState(mockScannedRobotEvent).getCurrentDistanceToEnemy(),
    //                StateName.DISTANCE_TO_ENEMY.MID);
    //    }
}
