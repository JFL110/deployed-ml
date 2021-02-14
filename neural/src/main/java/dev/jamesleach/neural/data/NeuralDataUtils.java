package dev.jamesleach.neural.data;

import lombok.AccessLevel;
import lombok.NonNull;
import lombok.RequiredArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Static collection of utils for data transformation.
 */
@RequiredArgsConstructor(access = AccessLevel.PRIVATE)
public class NeuralDataUtils {
  /**
   * index -> array of probabilities with one at the selected index and zeros at the others.
   */
  public static double[] toLabelProbabilityArray(int index, int numLabels) {
    double[] out = new double[numLabels];
    for (int i = 0; i < numLabels; i++) {
      out[i] = i == index ? 1 : 0;
    }
    return out;
  }


  /**
   * array of probabilities -> index of highest value in the array.
   */
  public static int highestProbabilityLabelIndex(@NonNull double[] labelProbabilities) {
    double maxValue = Double.MIN_VALUE;
    int index = -1;
    for (int i = 0; i < labelProbabilities.length; i++) {
      if (labelProbabilities[i] > maxValue) {
        maxValue = labelProbabilities[i];
        index = i;
      }
    }
    return index;
  }


  /**
   * double array -> reshaped INDArray with one input row.
   */
  public static INDArray toSingleInputArray(@NonNull DataPoint dataPoint) {
    var dataShape = dataPoint.getDataShape();
    var output = Nd4j.zeros(1, dataShape.getDepth(), dataShape.getHeight(), dataShape.getLength());
    output.putRow(0, inInputArrayRow(dataShape, dataPoint::getInputData3d));
    return output;
  }


  /**
   * double array -> reshaped INDArray.
   */
  public static INDArray inInputArrayRow(@NonNull DataShape dataShape, @NonNull NeuralDataUtils.ArrayAccessor3D data) {
    var output = Nd4j.zeros(dataShape.getDepth(), dataShape.getHeight(), dataShape.getLength());
    // Input
    // [ [ [a, b], [c, d], [e, f] ]
    //   [ [h, i], [j, k], [l, m] ]
    // to
    // [ [ [ a, c, e ],
    //     [ h, j, l ] ],
    //   [ [ b, d, f ],
    //     [ i, k, m ] ]]
    for (int d = 0; d < dataShape.getDepth(); d++) {
      for (int l = 0; l < dataShape.getLength(); l++) {
        for (int h = 0; h < dataShape.getHeight(); h++) {
          output.putScalar(d, h, l, data.value(h, l, d));
        }
      }
    }
    return output;
  }


  interface ArrayAccessor3D {
    double value(int height, int width, int depth);
  }
}
