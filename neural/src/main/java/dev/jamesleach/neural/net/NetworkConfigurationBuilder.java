package dev.jamesleach.neural.net;

import dev.jamesleach.neural.data.DataShape;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;

public interface NetworkConfigurationBuilder {

  ComputationGraphConfiguration build(CommonNetSpecification commonNetConfig, DataShape dataShape);

  default ComputationGraphConfiguration build(CommonNetSpecification.CommonNetSpecificationBuilder commonNetConfig,
                                              DataShape dataShape) {
    return build(commonNetConfig.build(), dataShape);
  }
}