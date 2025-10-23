"""Timeout protection utilities for FreeCiv parsers."""

import functools
import signal
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator, TypeVar

from absl import logging

F = TypeVar("F", bound=Callable[..., Any])


class TimeoutError(Exception):
    """Raised when an operation times out."""

    pass


@contextmanager
def timeout_protection(timeout_seconds: float) -> Generator[None, None, None]:
    """Context manager for protecting operations with timeout.

    Uses both signal-based interruption (on Unix) and threading for
    better protection against ReDoS attacks by actually interrupting
    hanging operations.

    Args:
      timeout_seconds: Maximum time to allow for the operation

    Raises:
      TimeoutError: If operation exceeds timeout

    Example:
      with timeout_protection(5.0):
          result = expensive_regex_operation(text)
    """
    if timeout_seconds <= 0:
        # No timeout protection
        yield
        return

    timed_out = threading.Event()
    timer = None
    original_handler = None

    def timeout_handler() -> None:
        timed_out.set()

    def signal_handler(signum, frame):
        """Signal handler for SIGALRM timeout."""
        raise TimeoutError(
            f"Operation interrupted after {timeout_seconds:.2f} seconds timeout"
        )

    try:
        # Use signal-based timeout on Unix systems for better interruption
        if hasattr(signal, "SIGALRM"):
            try:
                original_handler = signal.signal(signal.SIGALRM, signal_handler)
                signal.alarm(int(timeout_seconds + 0.5))  # Round up for signal
            except ValueError:
                # signal can only be used in main thread, fall back to timer only
                pass

        # Also use timer as backup and for non-Unix systems
        timer = threading.Timer(timeout_seconds, timeout_handler)
        timer.start()

        start_time = time.time()
        yield

        # Check if we timed out (for systems without signal support)
        if timed_out.is_set():
            elapsed = time.time() - start_time
            raise TimeoutError(
                f"Operation timed out after {elapsed:.2f} seconds "
                f"(limit: {timeout_seconds:.2f} seconds)"
            )

    finally:
        # Clean up signal handler
        if hasattr(signal, "SIGALRM") and original_handler is not None:
            try:
                signal.alarm(0)  # Cancel alarm
                signal.signal(signal.SIGALRM, original_handler)
            except ValueError:
                # Ignore errors in cleanup
                pass

        # Clean up timer
        if timer:
            timer.cancel()


def with_timeout(timeout_seconds: float) -> Callable[[F], F]:
    """Decorator to add timeout protection to functions.

    Args:
      timeout_seconds: Maximum time to allow for function execution

    Returns:
      Decorated function with timeout protection

    Example:
      @with_timeout(5.0)
      def regex_search(pattern, text):
          return pattern.search(text)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with timeout_protection(timeout_seconds):
                return func(*args, **kwargs)

        return wrapper

    return decorator


class TimeoutProtectedRegex:
    """Wrapper for regex operations with timeout protection."""

    def __init__(self, timeout_seconds: float = 5.0):
        """Initialize with timeout configuration.

        Args:
          timeout_seconds: Default timeout for regex operations
        """
        self.timeout_seconds = timeout_seconds
        self._compiled_patterns = {}  # Cache for compiled regex patterns

    def search(self, pattern, text: str, flags: int = 0) -> Any:
        """Protected regex search with timeout.

        Args:
          pattern: Compiled regex pattern or string pattern
          text: Text to search
          flags: Regex flags

        Returns:
          Match object or None

        Raises:
          TimeoutError: If search exceeds timeout
        """
        try:
            with timeout_protection(self.timeout_seconds):
                if hasattr(pattern, "search"):
                    # Already compiled pattern
                    return pattern.search(text)
                else:
                    # String pattern - use cache to avoid recompiling
                    pattern_key = (pattern, flags)
                    if pattern_key not in self._compiled_patterns:
                        import re

                        self._compiled_patterns[pattern_key] = re.compile(
                            pattern, flags
                        )
                    return self._compiled_patterns[pattern_key].search(text)
        except TimeoutError as e:
            logging.warning("Regex search timed out: %s", e)
            raise

    def findall(self, pattern, text: str, flags: int = 0) -> list[str]:
        """Protected regex findall with timeout.

        Args:
          pattern: Compiled regex pattern or string pattern
          text: Text to search
          flags: Regex flags

        Returns:
          List of matches

        Raises:
          TimeoutError: If search exceeds timeout
        """
        try:
            with timeout_protection(self.timeout_seconds):
                if hasattr(pattern, "findall"):
                    # Already compiled pattern
                    return pattern.findall(text)
                else:
                    # String pattern - use cache to avoid recompiling
                    pattern_key = (pattern, flags)
                    if pattern_key not in self._compiled_patterns:
                        import re

                        self._compiled_patterns[pattern_key] = re.compile(
                            pattern, flags
                        )
                    return self._compiled_patterns[pattern_key].findall(text)
        except TimeoutError as e:
            logging.warning("Regex findall timed out: %s", e)
            raise

    def sub(self, pattern, repl: str, text: str, count: int = 0, flags: int = 0) -> str:
        """Protected regex substitution with timeout.

        Args:
          pattern: Compiled regex pattern or string pattern
          repl: Replacement string
          text: Text to process
          count: Maximum number of substitutions
          flags: Regex flags

        Returns:
          Text with substitutions applied

        Raises:
          TimeoutError: If substitution exceeds timeout
        """
        try:
            with timeout_protection(self.timeout_seconds):
                if hasattr(pattern, "sub"):
                    # Already compiled pattern
                    return pattern.sub(repl, text, count)
                else:
                    # String pattern - use cache to avoid recompiling
                    pattern_key = (pattern, flags)
                    if pattern_key not in self._compiled_patterns:
                        import re

                        self._compiled_patterns[pattern_key] = re.compile(
                            pattern, flags
                        )
                    return self._compiled_patterns[pattern_key].sub(repl, text, count)
        except TimeoutError as e:
            logging.warning("Regex substitution timed out: %s", e)
            raise
