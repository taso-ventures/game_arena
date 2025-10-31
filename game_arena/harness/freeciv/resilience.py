"""Resilience patterns for error recovery and fault tolerance.

This module provides circuit breaker implementation and unified retry
logic with exponential backoff and jitter for preventing thundering herd.
"""

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Tuple, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitBreakerState(Enum):
  """States for the circuit breaker pattern."""
  CLOSED = "closed"  # Normal operation, requests pass through
  OPEN = "open"  # Failures detected, requests blocked
  HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryStrategy:
  """Configuration for retry behavior with exponential backoff.

  Attributes:
      max_attempts: Maximum number of retry attempts (including first try)
      base_delay: Initial delay in seconds before first retry
      backoff_multiplier: Multiplier for exponential backoff (e.g., 2.0 = double each time)
      max_delay: Maximum delay cap in seconds
      jitter: Tuple of (min_factor, max_factor) for random jitter (e.g., (0.1, 0.3) = 10-30%)
      retry_on: Tuple of exception types that should trigger retry
      timeout: Optional overall timeout for all retries (None = no timeout)
  """
  max_attempts: int
  base_delay: float
  backoff_multiplier: float
  max_delay: float
  jitter: Tuple[float, float]
  retry_on: Tuple[Type[Exception], ...]
  timeout: Optional[float] = None

  def __post_init__(self):
    """Validate configuration after initialization."""
    if self.max_attempts < 1:
      raise ValueError("max_attempts must be at least 1")
    if self.base_delay < 0:
      raise ValueError("base_delay cannot be negative")
    if self.backoff_multiplier <= 1.0:
      raise ValueError("backoff_multiplier must be > 1.0")
    if self.max_delay < self.base_delay:
      raise ValueError("max_delay must be >= base_delay")
    if not (0 <= self.jitter[0] <= self.jitter[1] <= 1):
      raise ValueError("jitter must be (min, max) where 0 <= min <= max <= 1")


def calculate_backoff_with_jitter(
    attempt: int,
    base_delay: float,
    multiplier: float,
    max_delay: float,
    jitter: Tuple[float, float]
) -> float:
  """Calculate exponential backoff delay with jitter.

  Args:
      attempt: Current attempt number (0-indexed)
      base_delay: Base delay in seconds
      multiplier: Exponential multiplier
      max_delay: Maximum delay cap
      jitter: (min_factor, max_factor) for random jitter

  Returns:
      Delay in seconds with jitter applied

  Example:
      >>> calculate_backoff_with_jitter(0, 2.0, 1.5, 30.0, (0.1, 0.3))
      2.4  # ~2.0 * (1 + random(0.1, 0.3))
      >>> calculate_backoff_with_jitter(3, 2.0, 1.5, 30.0, (0.1, 0.3))
      8.1  # ~2.0 * 1.5^3 * (1 + random(0.1, 0.3))
  """
  # Calculate exponential backoff
  delay = base_delay * (multiplier ** attempt)

  # Apply maximum delay cap
  delay = min(delay, max_delay)

  # Apply jitter to prevent thundering herd
  jitter_factor = random.uniform(jitter[0], jitter[1])
  delay = delay * (1 + jitter_factor)

  return delay


def with_retry(strategy: RetryStrategy) -> Callable:
  """Decorator for adding retry logic with exponential backoff to async functions.

  Args:
      strategy: RetryStrategy configuration

  Returns:
      Decorator function

  Example:
      >>> STATE_QUERY_RETRY = RetryStrategy(
      ...     max_attempts=3,
      ...     base_delay=2.0,
      ...     backoff_multiplier=1.5,
      ...     max_delay=10.0,
      ...     jitter=(0.1, 0.3),
      ...     retry_on=(RuntimeError, asyncio.TimeoutError)
      ... )
      >>>
      >>> @with_retry(STATE_QUERY_RETRY)
      ... async def get_state(...):
      ...     # Implementation
      ...     pass
  """
  def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
      start_time = time.time()

      for attempt in range(strategy.max_attempts):
        # Check if overall timeout exceeded
        if strategy.timeout:
          elapsed = time.time() - start_time
          if elapsed >= strategy.timeout:
            raise asyncio.TimeoutError(
                f"{func.__name__} exceeded overall timeout of "
                f"{strategy.timeout}s after {attempt} attempts"
            )

        try:
          # Execute the function
          return await func(*args, **kwargs)

        except strategy.retry_on as e:
          # Last attempt - re-raise the exception
          if attempt == strategy.max_attempts - 1:
            logger.error(
                f"{func.__name__} failed after {strategy.max_attempts} attempts: {e}"
            )
            raise

          # Calculate backoff delay
          delay = calculate_backoff_with_jitter(
              attempt,
              strategy.base_delay,
              strategy.backoff_multiplier,
              strategy.max_delay,
              strategy.jitter
          )

          logger.warning(
              f"{func.__name__} attempt {attempt + 1}/{strategy.max_attempts} "
              f"failed: {e}. Retrying in {delay:.2f}s..."
          )

          # Wait before next retry
          await asyncio.sleep(delay)

        except Exception as e:
          # Unexpected exception type - don't retry
          logger.error(
              f"{func.__name__} failed with unexpected exception "
              f"(not retrying): {type(e).__name__}: {e}"
          )
          raise

    return wrapper
  return decorator


class CircuitBreaker:
  """Circuit breaker implementation for fault tolerance.

  The circuit breaker prevents cascading failures by:
  1. CLOSED: Normal operation, track failures
  2. OPEN: After threshold failures, block requests
  3. HALF_OPEN: After recovery timeout, test if service recovered

  Attributes:
      failure_threshold: Number of failures before opening circuit
      recovery_timeout: Seconds to wait before trying recovery
      success_threshold: Successes needed to close circuit from half-open
  """

  def __init__(
      self,
      failure_threshold: int = 5,
      recovery_timeout: float = 60.0,
      success_threshold: int = 3
  ):
    """Initialize circuit breaker.

    Args:
        failure_threshold: Failures before opening circuit
        recovery_timeout: Seconds before attempting recovery
        success_threshold: Successes to close circuit
    """
    self.failure_threshold = failure_threshold
    self.recovery_timeout = recovery_timeout
    self.success_threshold = success_threshold

    self.state = CircuitBreakerState.CLOSED
    self.failure_count = 0
    self.success_count = 0
    self.last_failure_time: Optional[float] = None
    self.last_state_change: float = time.time()

  def record_success(self) -> None:
    """Record a successful operation."""
    if self.state == CircuitBreakerState.HALF_OPEN:
      self.success_count += 1
      logger.debug(
          f"Circuit breaker success in HALF_OPEN: "
          f"{self.success_count}/{self.success_threshold}"
      )

      if self.success_count >= self.success_threshold:
        self._transition_to_closed()

    elif self.state == CircuitBreakerState.CLOSED:
      # Reset failure count on success
      self.failure_count = 0

  def record_failure(self, error: Optional[Exception] = None) -> None:
    """Record a failed operation.

    Args:
        error: Optional exception that caused the failure
    """
    self.last_failure_time = time.time()

    if self.state == CircuitBreakerState.CLOSED:
      self.failure_count += 1
      logger.warning(
          f"Circuit breaker failure in CLOSED: "
          f"{self.failure_count}/{self.failure_threshold}"
      )

      if self.failure_count >= self.failure_threshold:
        self._transition_to_open()

    elif self.state == CircuitBreakerState.HALF_OPEN:
      # Failure during recovery - back to open
      logger.warning("Circuit breaker failure in HALF_OPEN, reopening")
      self._transition_to_open()

  def allow_request(self) -> bool:
    """Check if request should be allowed.

    Returns:
        True if request should proceed, False if blocked

    Side effects:
        May transition from OPEN to HALF_OPEN if recovery timeout passed
    """
    if self.state == CircuitBreakerState.CLOSED:
      return True

    if self.state == CircuitBreakerState.HALF_OPEN:
      return True

    if self.state == CircuitBreakerState.OPEN:
      # Check if recovery timeout has passed
      if self.last_failure_time:
        time_since_failure = time.time() - self.last_failure_time
        if time_since_failure >= self.recovery_timeout:
          self._transition_to_half_open()
          return True

      return False

    return False

  def _transition_to_closed(self) -> None:
    """Transition to CLOSED state."""
    logger.info("Circuit breaker transitioning to CLOSED")
    self.state = CircuitBreakerState.CLOSED
    self.failure_count = 0
    self.success_count = 0
    self.last_state_change = time.time()

  def _transition_to_open(self) -> None:
    """Transition to OPEN state."""
    logger.warning(
        f"Circuit breaker transitioning to OPEN after "
        f"{self.failure_count} failures"
    )
    self.state = CircuitBreakerState.OPEN
    self.success_count = 0
    self.last_state_change = time.time()

  def _transition_to_half_open(self) -> None:
    """Transition to HALF_OPEN state for recovery testing."""
    logger.info("Circuit breaker transitioning to HALF_OPEN for recovery test")
    self.state = CircuitBreakerState.HALF_OPEN
    self.failure_count = 0
    self.success_count = 0
    self.last_state_change = time.time()

  def get_stats(self) -> dict:
    """Get circuit breaker statistics.

    Returns:
        Dictionary with current state and counters
    """
    return {
        "state": self.state.value,
        "failure_count": self.failure_count,
        "success_count": self.success_count,
        "last_failure_time": self.last_failure_time,
        "last_state_change": self.last_state_change,
        "time_since_last_failure": (
            time.time() - self.last_failure_time
            if self.last_failure_time
            else None
        )
    }

  def reset(self) -> None:
    """Reset circuit breaker to initial state."""
    logger.info("Circuit breaker manually reset to CLOSED")
    self._transition_to_closed()


# Common retry strategies for FreeCiv operations

STATE_QUERY_RETRY = RetryStrategy(
    max_attempts=3,
    base_delay=2.0,
    backoff_multiplier=1.5,
    max_delay=10.0,
    jitter=(0.1, 0.3),
    retry_on=(RuntimeError, asyncio.TimeoutError),
    timeout=30.0
)

MESSAGE_SEND_RETRY = RetryStrategy(
    max_attempts=3,
    base_delay=1.0,
    backoff_multiplier=2.0,
    max_delay=60.0,
    jitter=(0.1, 0.3),
    retry_on=(asyncio.TimeoutError, ConnectionError),
    timeout=None
)

CONNECTION_RETRY = RetryStrategy(
    max_attempts=3,
    base_delay=2.0,
    backoff_multiplier=2.0,
    max_delay=60.0,
    jitter=(0.1, 0.3),
    retry_on=(ConnectionError, OSError),
    timeout=120.0
)
