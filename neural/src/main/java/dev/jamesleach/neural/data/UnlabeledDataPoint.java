package dev.jamesleach.neural.data;

import lombok.Data;

/**
 * Implementation of {@link DataPoint} without a label.
 */
@Data
public class UnlabeledDataPoint implements DataPoint {
  private final double[][][] inputData3d;
  private final DataShape dataShape;

  @Override
  public double getInputData3d(int height, int width, int depth) {
    return inputData3d[height][width][depth];
  }
}
