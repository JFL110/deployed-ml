package dev.jamesleach.web;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * /ping endpoint for checking the app is up.
 */
@RestController
class PingController {
  @GetMapping("/ping")
  String ping() {
    return "pong";
  }
}
