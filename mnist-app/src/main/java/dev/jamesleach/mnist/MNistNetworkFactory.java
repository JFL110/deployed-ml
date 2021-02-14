package dev.jamesleach.mnist;

import com.google.common.collect.Iterators;
import dev.jamesleach.neural.data.LabeledDataPoint;
import dev.jamesleach.neural.data.LabeledDataSet;
import dev.jamesleach.neural.data.LabeledDataSetCollection;
import dev.jamesleach.neural.data.NeuralDataUtils;
import dev.jamesleach.neural.net.*;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.springframework.stereotype.Service;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

/**
 * Train and save a network using the MNIST dataset.
 */
@Service
@RequiredArgsConstructor
@Slf4j
class MNistNetworkFactory {
  private static final int HEIGHT = 28;
  private static final int WIDTH = 28;
  private static final int MAX_VALUE = 255;
  private static final int NUM_DIGITS = 10;

  private final NetworkSaver saver;
  private final NetworkLoader loader;
  private final NetworkSerializer serializer;
  private final NetworkTrainerBuilder trainerBuilder;

  void createNetwork(String networkId, Duration timeToSpend, Path trainingDataCsv, Path testingDataCsv) {
    // Parse training data
    var trainingData = readDataFromCsv(trainingDataCsv);

    // Define network
    var netConfig = new MNistFeedForward().build(
      // Use default learning rate and dropout
      CommonNetSpecification.builder(),
      trainingData.getDataShape()
    );

    Consumer<ComputationGraph> save = n -> saver.save(
      new SerializedNetwork(networkId, trainingData.getDataShape(),
        serializer.serialize(n)));

    // Define network training regime
    var trainer = trainerBuilder.trainer(
      NetworkTrainerSpecification.builder()
        .initialModel(loader.load(networkId).map(n -> serializer.deserialize(n.getNetworkBinaryBase64())).orElse(null))
        .compNetworkConfiguration(netConfig)
        .trainingData(trainingData.toDataSetIterator())
        .maxTimeUnit(TimeUnit.SECONDS)
        .maxTime(timeToSpend.getSeconds())
        .bestModelSaver(save)
        .build());

    // Train
    log.info("Spending {} seconds on training", timeToSpend.getSeconds());
    var net = trainer.fit().getBestModel();

    // Evaluate
    log.info(net.evaluate(trainingData.toDataSetIterator()).toString());
    log.info(net.evaluate(readDataFromCsv(testingDataCsv).toDataSetIterator()).toString());

    save.accept(net);
  }


  /**
   * Read, parse and batch all CSV lines.
   */
  @SneakyThrows
  private LabeledDataSetCollection readDataFromCsv(Path csvPath) {
    log.info("Loading csv data...");
    return new LabeledDataSetCollection(
      StreamSupport.stream(Spliterators.spliteratorUnknownSize(
        Iterators.partition(
          Files.readAllLines(csvPath)
            .stream()
            // Skip the header row
            .skip(1)
            .parallel()
            // Parse
            .map(this::csvRowToDataPoint)
            // Chunk into batches
            .iterator(), 1000), Spliterator.NONNULL), false)
        // Batch -> DataSet
        .map(LabeledDataSet::new)
        .collect(Collectors.toList()));
  }


  /**
   * Single CSV row to a data point with label.
   */
  private LabeledDataPoint csvRowToDataPoint(String l) {
    String[] parts = l.split(",");
    int labelValue = Integer.parseInt(parts[0]);
    double[][][] input = new double[HEIGHT][WIDTH][1];
    for (int h = 0; h < HEIGHT; h++) {
      for (int w = 0; w < WIDTH; w++) {
        input[h][w][0] = Integer.parseInt(parts[1 + (w * WIDTH + h)]) / (double) MAX_VALUE;
      }
    }
    return new LabeledDataPoint(input, NeuralDataUtils.toLabelProbabilityArray(labelValue, NUM_DIGITS));
  }
}
