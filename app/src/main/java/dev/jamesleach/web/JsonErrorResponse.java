package dev.jamesleach.web;

import com.fasterxml.jackson.annotation.JsonCreator;
import lombok.Data;
import lombok.RequiredArgsConstructor;

/**
 * JSON format for error responses.
 */
@Data
@RequiredArgsConstructor(onConstructor = @__(@JsonCreator))
public class JsonErrorResponse {
  private final String message;
}
