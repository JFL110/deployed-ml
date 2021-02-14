package dev.jamesleach.neural.net;

/**
 * Save a network
 */
public interface NetworkSaver {
  /**
   * @param network the network to save
   */
  void save(SerializedNetwork network);
}
