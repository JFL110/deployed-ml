package dev.jamesleach.neural.net;

import lombok.Getter;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.nn.graph.ComputationGraph;

/**
 * EarlyStoppingGraphTrainer + NetworkTrainerSpecification
 */
@Getter
public class MetaEquipEarlyStoppingGraphTrainer extends EarlyStoppingGraphTrainer {
  private final NetworkTrainerSpecification specification;

  MetaEquipEarlyStoppingGraphTrainer(EarlyStoppingConfiguration<ComputationGraph> esConfig,
                                     ComputationGraph net,
                                     NetworkTrainerSpecification specification) {
    super(esConfig, net, specification.getTrainingData(), null);
    this.specification = specification;
  }
}