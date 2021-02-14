package dev.jamesleach.neural.data;

import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.adapter.SingletonDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

/**
 * A collection of labeled data points for training a neural network.
 */
@Data
public class LabeledDataSet {

  private final DataShape dataShape;
  private final DataSet dataSet;

  /**
   * Create a Deeplearning4j Dataset from a list of {@link LabeledDataPoint}.
   */
  public LabeledDataSet(List<LabeledDataPoint> data) {
    if (data.isEmpty()) {
      throw new IllegalStateException("Empty data");
    }

    dataShape = data.get(0).getDataShape();
    if (dataShape.getNumDimensions() != 3) {
      throw new IllegalStateException("Only 3d data implemented");
    }

    INDArray outputNDArray = Nd4j.zeros(data.size(), dataShape.getNumLabels());
    INDArray inputNDArray = Nd4j.zeros(data.size(), dataShape.getDepth(), dataShape.getHeight(), dataShape.getLength());
    for (int i = 0; i < data.size(); i++) {
      inputNDArray.putRow(i, NeuralDataUtils.inInputArrayRow(dataShape, data.get(i)::getInputData3d));
      outputNDArray.putRow(i, Nd4j.create(data.get(i).getLabels(), new int[]{dataShape.getNumLabels()}));
    }

    this.dataSet = new DataSet(inputNDArray, outputNDArray);
  }

  public SingletonDataSetIterator toSingletonIterator() {
    return new SingletonDataSetIterator(getDataSet());
  }
}
