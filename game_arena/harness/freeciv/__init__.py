"""FreeCiv integration modules for Game Arena.

This package provides configuration and resilience utilities for
FreeCiv proxy client operations.
"""

from game_arena.harness.freeciv.config import (
    CacheConfig,
    CircuitBreakerConfig,
    ConnectionConfig,
    DEFAULT_CONFIG,
    ProxyClientConfig,
    RateLimitConfig,
    RetryConfig,
    SecurityConfig,
)
from game_arena.harness.freeciv.resilience import (
    CONNECTION_RETRY,
    MESSAGE_SEND_RETRY,
    STATE_QUERY_RETRY,
    CircuitBreaker,
    CircuitBreakerState,
    RetryStrategy,
    calculate_backoff_with_jitter,
    with_retry,
)

__all__ = [
    # Configuration
    "CacheConfig",
    "CircuitBreakerConfig",
    "ConnectionConfig",
    "DEFAULT_CONFIG",
    "ProxyClientConfig",
    "RateLimitConfig",
    "RetryConfig",
    "SecurityConfig",
    # Resilience
    "CONNECTION_RETRY",
    "MESSAGE_SEND_RETRY",
    "STATE_QUERY_RETRY",
    "CircuitBreaker",
    "CircuitBreakerState",
    "RetryStrategy",
    "calculate_backoff_with_jitter",
    "with_retry",
]
