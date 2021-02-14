package dev.jamesleach.mnist;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.jamesleach.neural.data.ClassificationOutput;
import dev.jamesleach.neural.data.DataShape;
import dev.jamesleach.neural.data.UnlabeledDataPoint;
import dev.jamesleach.neural.net.SavedNetworkRunner;
import dev.jamesleach.web.BadRequestException;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.event.ContextStartedEvent;
import org.springframework.context.event.EventListener;
import org.springframework.core.io.ClassPathResource;
import org.springframework.scheduling.annotation.Async;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

/**
 * /classify-digit endpoint
 * Accept a pixel map, pass through a network and return the network prediction.
 */
@RestController
@Slf4j
@RequiredArgsConstructor
class DigitClassificationController {
  private static final int MAX_PIXEL_VALUE = 255;
  private static final DataShape MNIST_DATA_SHAPE = new DataShape(3, 9, 28, 28, 1);

  @Value("${network-id}")
  private final String networkId;
  private final ObjectMapper objectMapper;
  private final SavedNetworkRunner savedNetworkRunner;

  @PostMapping("/classify-digit")
  ClassificationOutput classifyDigit(@RequestBody DigitClassificationInput input) {
    // Validate
    if (input == null || input.getPixels() == null) {
      throw new BadRequestException("Null input");
    }
    if (input.getPixels().length != MNIST_DATA_SHAPE.getHeight()) {
      throw new BadRequestException("Height must be " + MNIST_DATA_SHAPE.getHeight() + " but got " + input.getPixels().length);
    }

    // Add the 3rd dimension
    double[][][] as3D = new double[MNIST_DATA_SHAPE.getHeight()][MNIST_DATA_SHAPE.getLength()][MNIST_DATA_SHAPE.getDepth()];
    for (int h = 0; h < MNIST_DATA_SHAPE.getHeight(); h++) {
      if (input.getPixels()[h].length != MNIST_DATA_SHAPE.getLength()) {
        throw new BadRequestException("Invalid length at row " + h + " should be " +
          MNIST_DATA_SHAPE.getLength() + " but got " + input.getPixels()[h].length);
      }
      for (int l = 0; l < MNIST_DATA_SHAPE.getLength(); l++) {
        as3D[h][l] = new double[]{input.getPixels()[h][l] / (double) MAX_PIXEL_VALUE};
      }
    }

    // Classify
    var point = new UnlabeledDataPoint(as3D, MNIST_DATA_SHAPE);
    return savedNetworkRunner.runClassification(networkId, point);
  }


  @Async
  @EventListener
  @SneakyThrows
  public void handleContextStart(ContextStartedEvent cse) {
    log.info("Running warmup point through network");
    var warmupPoint = objectMapper.readValue(
      new ClassPathResource("warmup-point.json").getInputStream(),
      DigitClassificationInput.class);
    classifyDigit(warmupPoint);
  }


  /**
   * JSON input format.
   */
  @Data
  @RequiredArgsConstructor(onConstructor = @__(@JsonCreator))
  static class DigitClassificationInput {
    private final double[][] pixels;
  }
}
