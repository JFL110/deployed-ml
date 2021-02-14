package dev.jamesleach;

import org.springframework.stereotype.Component;

import java.time.ZonedDateTime;
import java.util.function.Supplier;

/**
 * Inject the current time
 */
@Component
public class ZonedNowSupplier implements Supplier<ZonedDateTime> {
   @Override
   public ZonedDateTime get() {
      return ZonedDateTime.now();
   }
}
