package org.homework.rl;

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
}
