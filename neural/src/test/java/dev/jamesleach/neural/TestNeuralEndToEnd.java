package dev.jamesleach.neural;

import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableList;
import dev.jamesleach.neural.data.*;
import dev.jamesleach.neural.net.*;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.dropout.GaussianDropout;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Optional;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

@Slf4j
class TestNeuralEndToEnd {
  private static final String NET_ID = "net-id";
  private final NetworkLoader networkLoader = mock(NetworkLoader.class);
  private final NetworkTrainerBuilder trainerBuilder = new NetworkTrainerBuilder();
  private final NetworkSerializer networkSerializer = new NetworkSerializer();
  private final SavedNetworkRunner savedNetworkRunner = new SavedNetworkRunner(networkLoader, networkSerializer);

  @Test
  void test() {
    // Test data
    double[][][] pointData1 = {
      {{0.7}, {0.5}, {0.5}, {0.5}},
      {{0.7}, {0.5}, {0.5}, {0.5}}};
    double[][][] pointData2 = {
      {{0.5}, {0.75}, {0.5}, {0.5}},
      {{0.5}, {0.75}, {0.5}, {0.5}}};
    double[][][] pointData3 = {
      {{0.4}, {0.4}, {0.65}, {0.4}},
      {{0.4}, {0.4}, {0.65}, {0.4}}};
    double[][][] pointData4 = {
      {{0.5}, {0.5}, {0.5}, {0.8}},
      {{0.5}, {0.5}, {0.5}, {0.84}}};

    var point1 = new LabeledDataPoint(pointData1, new double[]{1, 0, 0, 0});
    var point2 = new LabeledDataPoint(pointData2, new double[]{0, 1, 0, 0});
    var point3 = new LabeledDataPoint(pointData3, new double[]{0, 0, 1, 0});
    var point4 = new LabeledDataPoint(pointData4, new double[]{0, 0, 0, 1});
    var dataSet = new LabeledDataSet(ImmutableList.of(point1, point2, point3, point4));

    // Network
    var conf = new FeedForwardNetwork().build(
      CommonNetSpecification.builder(),
      dataSet.getDataShape());

    var savedModelHolder = new AtomicReference<ComputationGraph>();
    var timer = Stopwatch.createStarted();
    var trainer = trainerBuilder.trainer(NetworkTrainerSpecification.builder()
      .compNetworkConfiguration(conf)
      .bestModelSaver(savedModelHolder::set)
      .maxTime(10L)
      .maxTimeUnit(TimeUnit.SECONDS)
      .trainingData(dataSet.toSingletonIterator())
      .build());

    // Train
    var bestModel = trainer.fit().getBestModel();
    assertTrue(timer.elapsed(TimeUnit.SECONDS) < 15);

    // Evaluate
    var evaluation = bestModel.evaluate(dataSet.toSingletonIterator());
    log.info(evaluation.toString());
    assertTrue(evaluation.accuracy() > 0.5);
    assertTrue(evaluation.precision() > 0.5);
    assertTrue(evaluation.recall() > 0.5);
    assertTrue(evaluation.f1() > 0.5);

    var pointOneOutput = bestModel.output(NeuralDataUtils.toSingleInputArray(point1))[0];
    assertArrayEquals(new long[]{1, 4}, pointOneOutput.shape());
    assertEquals(0, NeuralDataUtils.highestProbabilityLabelIndex(pointOneOutput.toDoubleVector()));

    // Runner
    when(networkLoader.load(NET_ID)).thenReturn(Optional.of(
      new SerializedNetwork(NET_ID, dataSet.getDataShape(), networkSerializer.serialize(bestModel))));

    var pointOneClassification = savedNetworkRunner.runClassification(NET_ID, point1);
    assertEquals(0, pointOneClassification.getLabelIndex());
    assertEquals(0, NeuralDataUtils.highestProbabilityLabelIndex(pointOneClassification.getLabelProbabilities()));

    assertEquals(1, savedNetworkRunner.runClassification(NET_ID, point2).getLabelIndex());
    assertEquals(2, savedNetworkRunner.runClassification(NET_ID, point3).getLabelIndex());
    assertEquals(3, savedNetworkRunner.runClassification(NET_ID, point4).getLabelIndex());

    var point5 = new UnlabeledDataPoint(new double[][][]{
      {{0.65}, {0.5}, {0.5}, {0.5}},
      {{0.76}, {0.5}, {0.5}, {0.5}}},
      dataSet.getDataShape());

    assertEquals(0, savedNetworkRunner.runClassification(NET_ID, point5).getLabelIndex());
  }


  static class FeedForwardNetwork implements NetworkConfigurationBuilder {
    @Override
    public ComputationGraphConfiguration build(CommonNetSpecification c, DataShape shape) {
      return new NeuralNetConfiguration.Builder()
        .seed(c.getSeed())
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .updater(new Adam(c.getLearningRate()))
        .graphBuilder()
        .addInputs("input")
        .setInputTypes(InputType.convolutional(
          shape.getHeight(),
          shape.getLength(),
          shape.getDepth()))
        .addLayer("l1",
          new DenseLayer.Builder()
            .activation(Activation.RELU)
            .weightInit(WeightInit.XAVIER)
            .dropOut(new GaussianDropout(c.getDropout()))
            .nOut(10)
            .build(),
          "input")
        .addLayer("output",
          new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
            .activation(Activation.SOFTMAX)
            .weightInit(WeightInit.XAVIER)
            .nOut(shape.getNumLabels())
            .dropOut(new GaussianDropout(c.getDropout()))
            .build(),
          "l1")
        .setOutputs("output")
        .build();
    }
  }
}
