package org.homework.rl;

import org.homework.robot.model.ImmutableState;
import org.homework.robot.model.State;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class LUTImplTest {

    @BeforeEach
    void setUp() {}

    @Test
    void TEST_INITIALIZE_LUT() {
        final LUTImpl lut = new LUTImpl(10, 3, 3, 2, .5, new int[] {1, 2}, new int[] {1, 2});

        assertArrayEquals(lut.getQTable().shape(), new long[] {3, 3, 2});
    }

    @Test
    void see_if_state_prints_anything() {
        final State state = ImmutableState.builder().build();

        System.out.println(state);
    }
}
