package dev.jamesleach.neural.data;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterators;
import org.junit.jupiter.api.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.hasItem;
import static org.hamcrest.Matchers.hasSize;
import static org.junit.jupiter.api.Assertions.*;


class TestDataSetDtos {

  @Test
  void testLabeledDataPointSingleDepth() {
    // Labeled data point
    double[][][] data = {
      {{1}, {2}, {3}},
      {{4}, {5}, {6}}
    };
    double[] label = {0, 0, 1, 0};
    var point = new LabeledDataPoint(data, label);

    assertEquals(data, point.getInputData3d());
    assertEquals(label, point.getLabels());
    assertEquals(3, point.getDataShape().getNumDimensions());
    assertEquals(2, point.getDataShape().getHeight());
    assertEquals(3, point.getDataShape().getLength());
    assertEquals(1, point.getDataShape().getDepth());
    assertEquals(4, point.getDataShape().getNumLabels());

    // Data set - single point
    var dataSet = new LabeledDataSet(ImmutableList.of(point));
    assertEquals(point.getDataShape(), dataSet.getDataShape());

    var features = dataSet.getDataSet().getFeatures();
    var labels = dataSet.getDataSet().getLabels();

    assertEquals(6, features.length());
    assertArrayEquals(new long[]{1, 1, 2, 3}, features.shape());

    assertEquals(4, labels.length());
    assertArrayEquals(new long[]{1, 4}, labels.shape());

    // Data set - multi point
    dataSet = new LabeledDataSet(ImmutableList.of(point, point));
    assertEquals(point.getDataShape(), dataSet.getDataShape());

    features = dataSet.getDataSet().getFeatures();
    labels = dataSet.getDataSet().getLabels();

    assertEquals(12, features.length());
    assertArrayEquals(new long[]{2, 1, 2, 3}, features.shape());

    assertEquals(8, labels.length());
    assertArrayEquals(new long[]{2, 4}, labels.shape());

    // Singleton iterator
    assertEquals(dataSet.getDataSet(), Iterators.getOnlyElement(dataSet.toSingletonIterator()));
  }


  @Test
  void testLabeledDataPointDualDepth() {
    double[][][] data = {
      {{1, 2}, {3, 4}, {5, 6}},
      {{7, 8}, {9, 10}, {11, 12}}
    };
    double[] label = {0, 1};
    var point = new LabeledDataPoint(data, label);

    assertEquals(data, point.getInputData3d());
    assertEquals(label, point.getLabels());
    assertEquals(3, point.getDataShape().getNumDimensions());
    assertEquals(2, point.getDataShape().getHeight());
    assertEquals(3, point.getDataShape().getLength());
    assertEquals(2, point.getDataShape().getDepth());
    assertEquals(2, point.getDataShape().getNumLabels());

    // Data set - single point
    var dataSet = new LabeledDataSet(ImmutableList.of(point));
    assertEquals(point.getDataShape(), dataSet.getDataShape());

    var features = dataSet.getDataSet().getFeatures();
    var labels = dataSet.getDataSet().getLabels();

    assertEquals(12, features.length());
    assertArrayEquals(new long[]{1, 2, 2, 3}, features.shape());

    assertEquals(2, labels.length());
    assertArrayEquals(new long[]{1, 2}, labels.shape());

    // Data set - multi point
    dataSet = new LabeledDataSet(ImmutableList.of(point, point));
    assertEquals(point.getDataShape(), dataSet.getDataShape());

    features = dataSet.getDataSet().getFeatures();
    labels = dataSet.getDataSet().getLabels();

    assertEquals(24, features.length());
    assertArrayEquals(new long[]{2, 2, 2, 3}, features.shape());

    assertEquals(4, labels.length());
    assertArrayEquals(new long[]{2, 2}, labels.shape());

    // Data set collection - single
    var dataSetCollection = new LabeledDataSetCollection(dataSet);
    assertEquals(point.getDataShape(), dataSetCollection.getDataShape());
    assertThat(dataSetCollection.getDataSets(), hasSize(1));
    assertThat(dataSetCollection.getDataSets(), hasItem(dataSet));

    assertEquals(dataSet.getDataSet(), Iterators.getOnlyElement(dataSetCollection.toDataSetIterator()));
  }


  @Test
  void testCannotCreateEmptyDataSet() {
    assertThrows(IllegalStateException.class, () -> new LabeledDataSet(ImmutableList.of()));
  }
}