package dev.jamesleach;

import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;

/**
 * Static utils related to Exceptions
 */
@RequiredArgsConstructor(access = AccessLevel.PRIVATE)
public class ExceptionUtils {

   /**
    * Run a runnable and re-throw an exception as a RuntimeException
    */
   public static void doRethrowing(ThrowingRunnable runnable) {
      try {
         runnable.run();
      } catch (Exception e) {
         throw new RuntimeException(e);
      }
   }

   /**
    * Call a supplier and re-throw an exception as a RuntimeException
    */
   public static <T> T doRethrowing(ThrowingSupplier<T> supplier) {
      try {
         return supplier.get();
      } catch (Exception e) {
         throw new RuntimeException(e);
      }
   }


   /**
    * {@link Runnable} but throws any exception
    */
   @FunctionalInterface
   public interface ThrowingRunnable {
      void run() throws Exception;
   }

   /**
    * {@link java.util.function.Supplier} but throws any exception
    */
   @FunctionalInterface
   public interface ThrowingSupplier<T> {
      T get() throws Exception;
   }
}