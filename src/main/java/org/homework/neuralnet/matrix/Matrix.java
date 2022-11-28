package org.homework.neuralnet.matrix;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.homework.neuralnet.matrix.op.ElementTransformation;
import org.homework.util.Util;

import java.util.Arrays;

@AllArgsConstructor
@NoArgsConstructor
@Getter
@Setter
public class Matrix {
    public int rowNum;
    public int colNum;
    public double[][] data;

    public Matrix(final double[][] _data) {
        this.rowNum = _data.length;
        this.colNum = _data[0].length;
        this.data = _data;
    }

    public Matrix(final int r, final int c) {
        this(new double[r][c]);
    }

    public static Matrix initZeroMatrix(final int r, final int c) {
        return new Matrix(r, c, new double[r][c]);
    }

    public static Matrix initRandMatrix(final int r, final int c) {
        return new Matrix(r, c).elementWiseOp(0, (a, b) -> Math.random());
    }

    /**
     * return a matrix which is the result of this matrix dot products with other matrix
     *
     * @param matrix matrix on the rhs
     * @return a matrix which is the result of this matrix dot products with other matrix
     */
    public Matrix mmul(final Matrix matrix) {
        final double[][] newData = new double[this.data.length][matrix.data[0].length];
        for (int i = 0; i < newData.length; i++) {
            for (int j = 0; j < newData[i].length; j++) {
                newData[i][j] = this.multiplyMatricesCell(this.data, matrix.data, i, j);
            }
        }
        return new Matrix(newData);
    }

    private double multiplyMatricesCell(
            final double[][] firstMatrix,
            final double[][] secondMatrix,
            final int row,
            final int col) {
        double cell = 0;
        for (int i = 0; i < secondMatrix.length; i++) {
            cell += firstMatrix[row][i] * secondMatrix[i][col];
        }
        return cell;
    }

    public Matrix transpose() {
        final double[][] temp = new double[this.data[0].length][this.data.length];
        for (int i = 0; i < this.rowNum; i++)
            for (int j = 0; j < this.colNum; j++) temp[j][i] = this.data[i][j];
        return new Matrix(temp);
    }

    public Matrix mul(final double x) {
        return this.elementWiseOp(x, (a, b) -> a * b);
    }

    public Matrix mul(final Matrix matrix) {
        return this.elementWiseOp(matrix, (a, b) -> a * b);
    }

    public Matrix add(final double x) {
        return this.elementWiseOp(x, Double::sum);
    }

    public Matrix add(final Matrix matrix) {
        return this.elementWiseOp(matrix, Double::sum);
    }

    public Matrix sub(final double x) {
        return this.elementWiseOp(x, (a, b) -> a - b);
    }

    public Matrix sub(final Matrix matrix) {
        return this.elementWiseOp(matrix, (a, b) -> a - b);
    }

    public Matrix elementWiseOp(
            final double operand, final ElementTransformation elementTransformation) {
        final double[][] dataClone = Util.getDeepArrayCopy(this.data);
        for (int i = 0; i < dataClone.length; i++) {
            for (int j = 0; j < dataClone[0].length; j++) {
                dataClone[i][j] = elementTransformation.op(dataClone[i][j], operand);
            }
        }
        return new Matrix(dataClone);
    }

    public Matrix elementWiseOp(
            final Matrix matrix, final ElementTransformation elementTransformation) {
        final double[][] dataClone = Util.getDeepArrayCopy(this.data);
        for (int i = 0; i < dataClone.length; i++) {
            for (int j = 0; j < dataClone[0].length; j++) {
                dataClone[i][j] = elementTransformation.op(dataClone[i][j], matrix.data[i][j]);
            }
        }
        return new Matrix(dataClone);
    }

    public double sumValues() {
        return Arrays.stream(this.data).flatMapToDouble(Arrays::stream).sum();
    }
}
