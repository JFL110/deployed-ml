package dev.jamesleach.neural.net;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.NonNull;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Optional;

@Component
@Slf4j
@RequiredArgsConstructor
class LocalFileNetworkSaverLoader implements NetworkLoader, NetworkSaver {
  private static final Charset CHARSET = StandardCharsets.UTF_8;
  private static final Path DIR = Paths.get("./saved-networks");

  private final ObjectMapper objectMapper;

  @SneakyThrows
  @Override
  public Optional<SerializedNetwork> load(@NonNull String id) {
    log.info("Loading network '{}'", id);
    initDir();
    if (!file(id).toFile().exists()) {
      log.info("No file '{}'", file(id));
      return Optional.empty();
    }
    String json = Files.readString(file(id));
    SerializedNetwork net = objectMapper.readValue(json, SerializedNetwork.class);
    return Optional.of(net);
  }

  @SneakyThrows
  @Override
  public void save(@NonNull SerializedNetwork network) {
    log.info("Saving network '{}'", network.getId());
    initDir();
    String json = objectMapper.writeValueAsString(network);
    Files.write(file(network.getId()), json.getBytes(CHARSET));
  }

  private Path file(String id) {
    return DIR.resolve(id + ".net" + SerializedNetwork.JSON_VERSION + ".json");
  }

  private void initDir() {
    if (!DIR.toFile().exists() && !DIR.toFile().mkdir())
      throw new IllegalStateException("Could not create working directory " + DIR);
  }
}