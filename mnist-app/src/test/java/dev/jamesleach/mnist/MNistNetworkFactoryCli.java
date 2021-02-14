package dev.jamesleach.mnist;

import lombok.RequiredArgsConstructor;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.WebApplicationType;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.context.annotation.Profile;

import java.nio.file.Paths;
import java.time.Duration;
import java.time.temporal.ChronoUnit;

/**
 * Run {@link MNistNetworkFactory} as a command line application.
 */
@Profile("mnist-cli")
@SpringBootApplication
@RequiredArgsConstructor
public class MNistNetworkFactoryCli implements CommandLineRunner {
  private final MNistNetworkFactory factory;

  /**
   * Run {@link MNistNetworkFactory} with no web context.
   */
  public static void main(String[] args) {
    new SpringApplicationBuilder(MNistNetworkFactoryCli.class)
      .profiles("mnist-cli")
      .web(WebApplicationType.NONE)
      .run(args);
  }

  @Override
  public void run(String... args) {
    factory.createNetwork(
      "feedforward-current",
      Duration.of(20, ChronoUnit.MINUTES),
      Paths.get("/home/jim/source/altcoin/neural/mnist_train.csv"),
      Paths.get("/home/jim/source/altcoin/neural/mnist_test.csv"));
  }
}
