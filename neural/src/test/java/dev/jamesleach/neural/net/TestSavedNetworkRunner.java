package dev.jamesleach.neural.net;

import dev.jamesleach.neural.data.DataShape;
import dev.jamesleach.neural.data.UnlabeledDataPoint;
import org.junit.Assert;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Optional;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class TestSavedNetworkRunner {
  private final NetworkLoader networkLoader = mock(NetworkLoader.class);
  private final NetworkSerializer networkSerializer = new NetworkSerializer();
  private final SavedNetworkRunner savedNetworkRunner = new SavedNetworkRunner(networkLoader, networkSerializer);

  @Test
  void noNetworkFound() {
    when(networkLoader.load("id")).thenReturn(Optional.empty());
    var exception = Assert.assertThrows(IllegalStateException.class, () ->
      savedNetworkRunner.runClassification("id",
        new UnlabeledDataPoint(
          new double[5][4][3],
          new DataShape(1, 1, 1, 1, 1))));

    Assertions.assertEquals("No network found for id 'id'", exception.getMessage());
  }
}
