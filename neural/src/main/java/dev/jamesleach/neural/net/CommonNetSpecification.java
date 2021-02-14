package dev.jamesleach.neural.net;

import lombok.Builder;
import lombok.Data;

/**
 * Hyper-parameters and configuration found on all networks.
 */
@Data
@Builder
public class CommonNetSpecification {
  /**
   * Default learning rate
   */
  public static final double DEFAULT_LEARNING_RATE = 0.005;

  @Builder.Default
  private final long seed = 1234;
  @Builder.Default
  private final double dropout = 0.2;
  @Builder.Default
  private final double learningRate = DEFAULT_LEARNING_RATE;
  @Builder.Default
  private final int sizeMultiplier = 1;
}
