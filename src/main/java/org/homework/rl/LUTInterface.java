package org.homework.rl;

import org.homework.CommonInterface;

/**
 * Interface for the Look Up Table (LUT) Class You should 'implement' this interface
 *
 * @date 20 June 012
 * @author sarbjit
 */
public interface LUTInterface extends CommonInterface {

    /**
     * Constructor. (You will need to define one in your implementation)
     *
     * @param argNumInputs The number of inputs in your input vector
     * @param argVariableFloor An array specifying the lowest value of each variable in the input
     *     vector.
     * @param argVariableCeiling An array specifying the highest value of each of the variables in
     *     the input vector. The order must match the order as referred to in argVariableFloor.
     *     <p>public LUT ( int argNumInputs, int [] argVariableFloor, int [] argVariableCeiling );
     */

    /** Initialise the look up table to all zeros. */
    void initialiseLUT();

    /**
     * A helper method that translates a vector being used to index the look up table into an
     * ordinal that can then be used to access the associated look up table element.
     *
     * @param X The state action vector used to index the LUT
     * @return The index where this vector maps to
     */
    int indexFor(double[] X);
} // End of public interface LUT
