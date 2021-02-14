package dev.jamesleach.neural.net;

import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.springframework.stereotype.Component;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.Base64;

/**
 * Serialize and Deserialize {@link ComputationGraph}s
 */
@Component
@Slf4j
public class NetworkSerializer {
  private static final Charset CHARSET = StandardCharsets.UTF_8;

  /**
   * ComputationGraph -> Base64 String
   */
  @SneakyThrows
  public String serialize(ComputationGraph graph) {
    var bos = new ByteArrayOutputStream();
    ModelSerializer.writeModel(graph, bos, true);
    return new String(Base64.getEncoder().encode(bos.toByteArray()), CHARSET);
  }

  /**
   * Base64 String -> ComputationGraph
   */
  @SneakyThrows
  public ComputationGraph deserialize(String base64) {
    byte[] bytes = Base64.getDecoder().decode(base64.getBytes(StandardCharsets.UTF_8));
    var bis = new ByteArrayInputStream(bytes);
    return ModelSerializer.restoreComputationGraph(bis);
  }
}
