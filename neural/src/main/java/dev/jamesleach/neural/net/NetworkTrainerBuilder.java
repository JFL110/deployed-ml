package dev.jamesleach.neural.net;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.scorecalc.ScoreCalculator;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.function.Consumer;

/**
 * Build an {@link EarlyStoppingGraphTrainer} to train a {@link ComputationGraph}
 */
@Service
@Slf4j
public class NetworkTrainerBuilder {
  /**
   * @param spec the configuration to use to build an {@link EarlyStoppingGraphTrainer}
   * @return an {@link EarlyStoppingGraphTrainer} to train a network
   */
  public MetaEquipEarlyStoppingGraphTrainer trainer(NetworkTrainerSpecification spec) {
    // Validate
    if (spec.getCompNetworkConfiguration() == null && spec.getInitialModel() == null) {
      throw new IllegalArgumentException("Must specify one of 'compNetworkConfiguration' or 'initialModel'");
    }

    // Trainer parts
    var saver = spec.getBestModelSaver() == null
      ? new InMemoryModelSaver<ComputationGraph>()
      : new DelegatingModelSaver(spec.getBestModelSaver());

    var scoreCalculator = spec.getEvaluationData() == null
      ? new CurrentScore<>(true)
      : new DataSetLossCalculator(spec.getEvaluationData(), true);

    // Network parts
    var net = spec.getInitialModel() == null
      ? new ComputationGraph(spec.getCompNetworkConfiguration())
      : spec.getInitialModel();

    net.init();
    log.info("Network has {} parameters", net.numParams());
    log.info(net.summary());

    return new MetaEquipEarlyStoppingGraphTrainer(
      new EarlyStoppingConfiguration.Builder<ComputationGraph>()
        .iterationTerminationConditions(
          new MaxTimeIterationTerminationCondition(spec.getMaxTime(), spec.getMaxTimeUnit()))
        .scoreCalculator(scoreCalculator)
        .evaluateEveryNEpochs(1)
        .modelSaver(saver)
        .build(),
      net,
      spec);
  }

  /**
   * ScoreCalculator that uses the last score on the network
   */
  @RequiredArgsConstructor
  private static class CurrentScore<T extends Model> implements ScoreCalculator<T> {
    private final boolean minimizeScore;

    @Override
    public double calculateScore(T network) {
      return network.score();
    }

    @Override
    public boolean minimizeScore() {
      return minimizeScore;
    }
  }


  @RequiredArgsConstructor
  private static class DelegatingModelSaver extends InMemoryModelSaver<ComputationGraph> {
    private final Consumer<ComputationGraph> delegateBestModelSaver;

    @Override
    public void saveBestModel(ComputationGraph net, double score) throws IOException {
      super.saveBestModel(net, score);
      delegateBestModelSaver.accept(net);
    }
  }
}
