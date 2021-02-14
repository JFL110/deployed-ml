package dev.jamesleach.web;

import org.springframework.http.HttpStatus;
import org.springframework.web.client.HttpClientErrorException;

/**
 * Exception for a 400.
 */
public class BadRequestException extends HttpClientErrorException {
  public BadRequestException(String message) {
    super(HttpStatus.BAD_REQUEST, message);
  }
}