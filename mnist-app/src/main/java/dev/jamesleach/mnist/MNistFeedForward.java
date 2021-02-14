package dev.jamesleach.mnist;

import dev.jamesleach.neural.data.DataShape;
import dev.jamesleach.neural.net.CommonNetSpecification;
import dev.jamesleach.neural.net.NetworkConfigurationBuilder;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.dropout.GaussianDropout;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Very simple feed-forward network accepting a convolution-style input.
 */
@Slf4j
class MNistFeedForward implements NetworkConfigurationBuilder {
  @Override
  public ComputationGraphConfiguration build(CommonNetSpecification c, DataShape shape) {
    return new NeuralNetConfiguration.Builder()
      .seed(c.getSeed())
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(new Adam(c.getLearningRate()))
      .graphBuilder()
      .addInputs("img1")
      .setInputTypes(InputType.convolutional(
        shape.getHeight(),
        shape.getLength(),
        shape.getDepth()))
      .addLayer("d1",
        new DenseLayer.Builder()
          .activation(Activation.RELU)
          .weightInit(WeightInit.XAVIER)
          .dropOut(new GaussianDropout(c.getDropout()))
          .nOut(200)
          .build(),
        "img1")
      .addLayer("output",
        new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
          .activation(Activation.SOFTMAX)
          .weightInit(WeightInit.XAVIER)
          .nOut(shape.getNumLabels())
          .dropOut(new GaussianDropout(c.getDropout()))
          .build(),
        "d1")
      .setOutputs("output")
      .build();
  }
}
