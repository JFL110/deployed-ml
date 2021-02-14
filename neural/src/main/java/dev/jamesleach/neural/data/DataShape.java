package dev.jamesleach.neural.data;

import com.fasterxml.jackson.annotation.JsonCreator;
import lombok.Data;
import lombok.RequiredArgsConstructor;

/**
 * Description of the shape of input data to a neural network.
 */
@Data
@RequiredArgsConstructor(onConstructor = @__(@JsonCreator))
public class DataShape {
   private final int numDimensions;
   private final int numLabels;
   private final int length;
   private final int height;
   private final int depth;
}
