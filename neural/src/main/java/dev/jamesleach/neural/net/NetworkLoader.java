package dev.jamesleach.neural.net;

import java.util.Optional;

/**
 * Load saved networks
 */
public interface NetworkLoader {
  /**
   * @param id the ID of the network to load
   * @return the saved network or Optional.empty() if no such network is found
   */
  Optional<SerializedNetwork> load(String id);
}
