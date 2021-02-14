package dev.jamesleach.web;

import org.springframework.http.HttpStatus;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.ResponseStatus;

/**
 * Exception -> JsonErrorResponse
 */
@ControllerAdvice
class InvalidInputExceptionMapper {
  @ExceptionHandler(BadRequestException.class)
  @ResponseStatus(HttpStatus.BAD_REQUEST)
  @ResponseBody
  JsonErrorResponse invalidInput(BadRequestException e) {
    return new JsonErrorResponse(e.getMessage());
  }

  @ExceptionHandler(HttpMessageNotReadableException.class)
  @ResponseStatus(HttpStatus.BAD_REQUEST)
  @ResponseBody
  JsonErrorResponse messageNotReadable(HttpMessageNotReadableException e) {
    return new JsonErrorResponse(e.getMessage());
  }
}