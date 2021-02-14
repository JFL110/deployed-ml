package dev.jamesleach.neural.data;

import lombok.Data;
import lombok.NonNull;

/**
 * Implementation of {@link DataPoint} with a label value for training.
 */
@Data
public class LabeledDataPoint implements DataPoint {
  /**
   * [height][length][depth]
   */
  @NonNull
  private final double[][][] inputData3d;
  @NonNull
  private final double[] labels;

  @Override
  public DataShape getDataShape() {
    int height = inputData3d.length;
    int length = height == 0 ? 0 : inputData3d[0].length;
    int depth = length == 0 ? 0 : inputData3d[0][0].length;
    int numLabels = labels.length;

    return new DataShape(
      3,
      numLabels,
      length,
      height,
      depth
    );
  }

  @Override
  public double getInputData3d(int height, int width, int depth) {
    return inputData3d[height][width][depth];
  }
}
