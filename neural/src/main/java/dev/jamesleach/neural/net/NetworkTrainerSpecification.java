package dev.jamesleach.neural.net;

import lombok.Builder;
import lombok.Data;
import lombok.NonNull;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/**
 * Configuration for {@link NetworkTrainerBuilder}
 */
@Data
@Builder
public class NetworkTrainerSpecification {
  private final ComputationGraph initialModel;
  private final ComputationGraphConfiguration compNetworkConfiguration;
  @NonNull
  private final Long maxTime;
  @NonNull
  private final TimeUnit maxTimeUnit;
  @NonNull
  private final DataSetIterator trainingData;
  private final DataSetIterator evaluationData;
  private final Consumer<ComputationGraph> bestModelSaver;
}