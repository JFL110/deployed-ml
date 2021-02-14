package dev.jamesleach.mnist;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication(scanBasePackages = "dev.jamesleach")
public class MNistRestApp {
  public static void main(String[] args) {
    SpringApplication.run(MNistRestApp.class, args).start();
  }
}
