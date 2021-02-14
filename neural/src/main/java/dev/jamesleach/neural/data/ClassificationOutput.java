package dev.jamesleach.neural.data;

import lombok.Data;

/**
 * Network output to a classification problem.
 */
@Data
public class ClassificationOutput {
  private final double[] labelProbabilities;
  private final int labelIndex;
}
