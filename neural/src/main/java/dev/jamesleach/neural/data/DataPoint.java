package dev.jamesleach.neural.data;

public interface DataPoint {
  /**
   * Network input in the form [height][width][depth]
   */
  double getInputData3d(int height, int width, int depth);

  /**
   * The shape of the network input data.
   */
  DataShape getDataShape();
}
