package org.homework.rl;

import org.homework.robot.model.ImmutableState;
import org.homework.robot.model.State;
import org.homework.robot.model.StateName;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LUTImplTest {

    @BeforeEach
    void setUp() {}

    @Test
    void TEST_GET_MAX_INDEX_FROM_INDARRAY() {
        final INDArray qTable = Nd4j.zeros(3, 3, 3);
        qTable.putScalar(new int[] {2, 2, 1}, 1);
        final int[] dimension = new int[] {2, 2};
        INDArray qValues = qTable.get(NDArrayIndex.point(dimension[0]));
        for (int i = 1; i < dimension.length; ++i) {
            qValues = qValues.get(NDArrayIndex.point(dimension[i]));
        }
        assertEquals(qValues.argMax().getInt(), 1);
    }

    @Test
    void TEST_CHOOSE_GREEDY_ACTION() {
        final State state = ImmutableState.builder().build();
        final LUTImpl lut = new LUTImpl(state);
        final State newState =
                ImmutableState.builder().from(state).currentHP(StateName.HP.LOW).build();
        assertTrue(lut.chooseGreedyAction(newState) < 5);
        System.out.println("hello");
    }

    @Test
    void see_if_state_prints_anything() {
        final State state = ImmutableState.builder().build();

        System.out.println(state);
    }
}
