package dev.jamesleach.neural.net;

import com.fasterxml.jackson.annotation.JsonCreator;
import dev.jamesleach.neural.data.DataShape;
import lombok.Data;
import lombok.RequiredArgsConstructor;

/**
 * A serialized {@link org.deeplearning4j.nn.graph.ComputationGraph} + meta data.
 */
@Data
@RequiredArgsConstructor(onConstructor = @__(@JsonCreator))
public class SerializedNetwork {
  /**
   * Version used to prevent parsing of old saved JSON.
   */
  public static final int JSON_VERSION = 1;

  private final String id;
  private final DataShape dataShape;
  private final String networkBinaryBase64;
}