package dev.jamesleach.mnist;

import com.fasterxml.jackson.databind.ObjectMapper;
import dev.jamesleach.mnist.DigitClassificationController.DigitClassificationInput;
import dev.jamesleach.neural.data.ClassificationOutput;
import dev.jamesleach.neural.net.NetworkLoader;
import dev.jamesleach.web.JsonErrorResponse;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.reactive.AutoConfigureWebTestClient;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.TestPropertySource;
import org.springframework.test.web.reactive.server.WebTestClient;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.temporal.ChronoUnit;
import java.util.Objects;

import static org.hamcrest.CoreMatchers.containsString;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.jupiter.api.Assertions.*;

/**
 * End to end test of the /mnist project.
 * Train a network on a tiny selection of the MNIST dataset.
 * Input one of the trained values to the endpoint and expect to get the correct digit back.
 */
@ActiveProfiles("test")
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@RequiredArgsConstructor(onConstructor = @__(@Autowired))
@TestPropertySource(properties = "network-id=test-network-id")
@AutoConfigureWebTestClient
@Slf4j
class TestEndToEnd {
  private static final String TEST_NETWORK_ID = "test-network-id";

  private final ObjectMapper objectMapper;
  private final MNistNetworkFactory mNistNetworkFactory;
  private final NetworkLoader networkLoader;
  private final WebTestClient webTestClient;


  @Test
  void testCorrectClassifications() throws Exception {
    // Create network
    mNistNetworkFactory.createNetwork(
      TEST_NETWORK_ID,
      Duration.of(2, ChronoUnit.SECONDS),
      localResourcesPath("./mnist-ten-rows-train.csv"),
      localResourcesPath("./mnist-ten-rows-test.csv")
    );

    // Verify created network
    var savedNetwork = networkLoader.load(TEST_NETWORK_ID).orElse(null);
    assertNotNull(savedNetwork);
    assertNotNull(savedNetwork.getNetworkBinaryBase64());
    assertEquals(TEST_NETWORK_ID, savedNetwork.getId());
    assertEquals(1, savedNetwork.getDataShape().getDepth());
    assertEquals(28, savedNetwork.getDataShape().getLength());
    assertEquals(28, savedNetwork.getDataShape().getHeight());
    assertEquals(3, savedNetwork.getDataShape().getNumDimensions());
    assertEquals(10, savedNetwork.getDataShape().getNumLabels());

    var inputFive = objectMapper.readValue(
      localResourcesPath("test-input-five.json").toFile(),
      DigitClassificationInput.class);

    var response = webTestClient
      .post()
      .uri("/classify-digit")
      .bodyValue(inputFive)
      .exchange()
      .expectStatus().isOk()
      .expectBody(ClassificationOutput.class)
      .returnResult()
      .getResponseBody();

    assertNotNull(response);
    assertEquals(5, response.getLabelIndex());
    assertTrue(response.getLabelProbabilities()[5] > 0.5);

    var inputOne = objectMapper.readValue(
      localResourcesPath("test-input-one.json").toFile(),
      DigitClassificationInput.class);

    response = webTestClient
      .post()
      .uri("/classify-digit")
      .bodyValue(inputOne)
      .exchange()
      .expectStatus().isOk()
      .expectBody(ClassificationOutput.class)
      .returnResult()
      .getResponseBody();

    assertNotNull(response);
    assertEquals(1, response.getLabelIndex());
    assertTrue(response.getLabelProbabilities()[1] > 0.5);
  }


  @Test
  public void testInvalidInputs() {
    // Null input
    webTestClient
      .post()
      .uri("/classify-digit")
      .exchange()
      .expectStatus().is4xxClientError()
      .expectBody(JsonErrorResponse.class)
      .consumeWith(r -> {
        assertNotNull(r.getResponseBody());
        assertThat(
          r.getResponseBody().getMessage(),
          containsString("Required request body is missing"));
      });

    // Null array input
    webTestClient
      .post()
      .uri("/classify-digit")
      .bodyValue(new DigitClassificationInput(null))
      .exchange()
      .expectStatus().is4xxClientError()
      .expectBody(JsonErrorResponse.class)
      .isEqualTo(new JsonErrorResponse("400 Null input"));

    // Incorrect width input
    webTestClient
      .post()
      .uri("/classify-digit")
      .bodyValue(new DigitClassificationInput(new double[28][27]))
      .exchange()
      .expectStatus().is4xxClientError()
      .expectBody(JsonErrorResponse.class)
      .isEqualTo(new JsonErrorResponse("400 Invalid length at row 0 should be 28 but got 27"));

    // Incorrect height input
    webTestClient
      .post()
      .uri("/classify-digit")
      .bodyValue(new DigitClassificationInput(new double[27][28]))
      .exchange()
      .expectStatus().is4xxClientError()
      .expectBody(JsonErrorResponse.class)
      .isEqualTo(new JsonErrorResponse("400 Height must be 28 but got 27"));
  }


  @Test
  void testPing(){
    webTestClient
      .get()
      .uri("/ping")
      .exchange()
      .expectStatus().isOk()
      .expectBody(String.class)
      .isEqualTo("pong");
  }


  @SneakyThrows
  private Path localResourcesPath(String fileName) {
    return Paths.get(Objects.requireNonNull(getClass().getClassLoader().getResource(fileName)).toURI());
  }
}
