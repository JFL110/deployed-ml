package dev.jamesleach.neural.net;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import dev.jamesleach.neural.data.ClassificationOutput;
import dev.jamesleach.neural.data.DataPoint;
import dev.jamesleach.neural.data.NeuralDataUtils;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.springframework.stereotype.Component;

/**
 * Run a single data point through a saved network.
 * Hard-cache the network based on ID.
 */
@Component
@RequiredArgsConstructor
public class SavedNetworkRunner {
  private static final long MAX_CACHED_NETWORKS = 100;
  private final NetworkLoader loader;
  private final NetworkSerializer serializer;

  private final LoadingCache<String, UnpackedNetwork> networkCache = CacheBuilder.newBuilder()
    .maximumSize(MAX_CACHED_NETWORKS)
    .build(CacheLoader.from(this::loadNetwork));


  /**
   * Run a single data point through a saved network.
   */
  public ClassificationOutput runClassification(String networkId, DataPoint dataPoint) {
    var network = networkCache.getUnchecked(networkId);
    var input = NeuralDataUtils.toSingleInputArray(dataPoint);
    var output = network.getNetwork().output(input)[0].toDoubleVector(); // Only one output supported
    int predictedIndex = NeuralDataUtils.highestProbabilityLabelIndex(output);
    return new ClassificationOutput(
      output,
      predictedIndex
    );
  }

  private UnpackedNetwork loadNetwork(String id) {
    var serializedNetwork = loader.load(id)
      .orElseThrow(() -> new IllegalStateException("No network found for id '" + id + "'"));
    var graph = serializer.deserialize(serializedNetwork.getNetworkBinaryBase64());
    return new UnpackedNetwork(serializedNetwork, graph);
  }

  @Data
  private static class UnpackedNetwork {
    private final SerializedNetwork serializedNetwork;
    private final ComputationGraph network;
  }
}
