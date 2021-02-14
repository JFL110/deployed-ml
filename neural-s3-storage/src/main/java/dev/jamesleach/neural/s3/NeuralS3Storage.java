package dev.jamesleach.neural.s3;

import com.amazonaws.auth.AWSCredentials;
import com.amazonaws.auth.AWSStaticCredentialsProvider;
import com.amazonaws.auth.BasicAWSCredentials;
import com.amazonaws.auth.DefaultAWSCredentialsProviderChain;
import com.amazonaws.regions.Regions;
import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.AmazonS3ClientBuilder;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.jamesleach.neural.net.NetworkLoader;
import dev.jamesleach.neural.net.NetworkSaver;
import dev.jamesleach.neural.net.SerializedNetwork;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Primary;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

import java.nio.charset.StandardCharsets;
import java.util.Optional;

@Profile("!test")
@Primary
@Slf4j
@Component
class NeuralS3Storage implements NetworkLoader, NetworkSaver {
  private static final Regions REGION = Regions.EU_WEST_2;
  private static final String BUCKET_NAME = "saved-networks";

  private final ObjectMapper objectMapper;
  private final AWSCredentials awSCredentials;

  NeuralS3Storage(ObjectMapper objectMapper,
                         @Value("${aws-access-key-id:}") String accessKeyId,
                         @Value("${aws-secret-access-key:}") String secretAccessKey) {
    this.objectMapper = objectMapper;
    if (StringUtils.isBlank(accessKeyId) || StringUtils.isBlank(secretAccessKey)) {
      log.info("Using DefaultAWSCredentialsProviderChain");
      awSCredentials = new DefaultAWSCredentialsProviderChain().getCredentials();
    } else {
      // Print the last 4 letters of the key
      log.info("Using AWS Access Key {}", accessKeyId.replaceAll(".(?=.{4})", "X"));
      awSCredentials = new BasicAWSCredentials(accessKeyId, secretAccessKey);
    }
  }

  @Override
  public Optional<SerializedNetwork> load(@NonNull String id) {
    try {
      var obj = client().getObject(BUCKET_NAME, objectName(id));
      if (obj == null) {
        return Optional.empty();
      }
      String contentString = IOUtils.toString(obj.getObjectContent(), StandardCharsets.UTF_8);
      return Optional.of(objectMapper.readValue(contentString, SerializedNetwork.class));
    } catch (Exception e) {
      log.error("Error loading network from S3", e);
      return Optional.empty();
    }
  }


  /**
   * putObject uses UTF-8 encoding.
   */
  @Override
  public void save(@NonNull SerializedNetwork network) {
    log.info("Saving network '{}' in S3", network.getId());
    try {
      String json = objectMapper.writeValueAsString(network);
      client().putObject(BUCKET_NAME, objectName(network.getId()), json);
    } catch (Exception e) {
      throw new RuntimeException("Error saving network in S3", e);
    }
  }

  private AmazonS3 client() {
    return AmazonS3ClientBuilder.standard()
      .withCredentials(new AWSStaticCredentialsProvider(awSCredentials))
      .withRegion(REGION).build();
  }

  private String objectName(String id) {
    return id + ".json";
  }
}
