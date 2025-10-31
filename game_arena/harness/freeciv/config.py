"""Configuration constants and settings for FreeCiv proxy client.

This module consolidates all magic numbers and configuration parameters
into typed dataclasses for better maintainability and configurability.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Pattern


@dataclass
class SecurityConfig:
  """Security-related configuration limits.

  These limits protect against DoS attacks, memory exhaustion,
  and other security vulnerabilities.
  """
  # JSON message limits
  max_json_size: int = 5_000_000  # 5MB limit for JSON messages
  max_json_depth: int = 100  # Maximum nesting depth to prevent DoS

  # WebSocket limits
  max_websocket_size: int = 5_000_000  # 5MB limit for WebSocket messages
  max_websocket_queue: int = 32  # Maximum queued messages
  websocket_close_timeout: float = 5.0  # Timeout for close operations

  # Input validation ranges
  max_action_id: int = 999999  # Maximum valid action ID
  min_action_id: int = 0  # Minimum valid action ID
  max_player_id: int = 16  # Maximum number of players in FreeCiv
  min_player_id: int = 0  # Minimum player ID
  max_observation_size: int = 5_000_000  # 5MB limit for observation data
  max_string_length: int = 10000  # Maximum length for string fields
  max_array_length: int = 10000  # Maximum length for arrays

  # Parser security
  max_target_string_length: int = 200  # Max length for Python repr parsing

  # Cache key validation pattern
  cache_key_pattern: Pattern = field(
      default_factory=lambda: re.compile(r'^[a-zA-Z0-9_.-]+$')
  )

  @classmethod
  def from_env(cls) -> 'SecurityConfig':
    """Create configuration from environment variables."""
    return cls(
        max_json_size=int(os.getenv('FREECIV_MAX_JSON_SIZE', '5000000')),
        max_json_depth=int(os.getenv('FREECIV_MAX_JSON_DEPTH', '100')),
        max_websocket_size=int(os.getenv('FREECIV_MAX_WEBSOCKET_SIZE', '5000000')),
        max_player_id=int(os.getenv('FREECIV_MAX_PLAYER_ID', '16')),
    )


@dataclass
class CircuitBreakerConfig:
  """Circuit breaker configuration for fault tolerance.

  The circuit breaker pattern prevents cascading failures by
  opening the circuit after repeated failures and closing it
  after successful recovery.
  """
  failure_threshold: int = 5  # Failures before opening circuit
  recovery_timeout: float = 60.0  # Seconds to wait before retry
  success_threshold: int = 3  # Successes needed to close circuit

  @classmethod
  def from_env(cls) -> 'CircuitBreakerConfig':
    """Create configuration from environment variables."""
    return cls(
        failure_threshold=int(os.getenv('FREECIV_CB_FAILURE_THRESHOLD', '5')),
        recovery_timeout=float(os.getenv('FREECIV_CB_RECOVERY_TIMEOUT', '60.0')),
        success_threshold=int(os.getenv('FREECIV_CB_SUCCESS_THRESHOLD', '3')),
    )


@dataclass
class RateLimitConfig:
  """Rate limiting configuration to prevent abuse.

  Implements token bucket rate limiting for both request count
  and bandwidth usage.
  """
  # Request rate limiting
  requests_per_minute: int = 60  # Requests per minute per player
  burst_size: int = 10  # Allow burst of N requests

  # Message frequency limits
  max_messages_per_second: int = 10  # Messages per second per connection
  max_bytes_per_minute: int = 50_000_000  # 50MB per minute per connection
  window_seconds: float = 60.0  # Rolling window for tracking

  @classmethod
  def from_env(cls) -> 'RateLimitConfig':
    """Create configuration from environment variables."""
    return cls(
        requests_per_minute=int(os.getenv('FREECIV_RATE_LIMIT_RPM', '60')),
        burst_size=int(os.getenv('FREECIV_RATE_LIMIT_BURST', '10')),
        max_messages_per_second=int(os.getenv('FREECIV_MAX_MSG_PER_SEC', '10')),
    )


@dataclass
class RetryConfig:
  """Retry and backoff configuration.

  Implements exponential backoff with jitter to prevent
  thundering herd problems.
  """
  # Retry parameters
  max_retries: int = 3  # Maximum retry attempts
  base_delay: float = 2.0  # Initial delay in seconds

  # Exponential backoff
  backoff_base: float = 1.5  # Multiplier for exponential backoff
  max_backoff_delay: float = 30.0  # Maximum backoff delay in seconds

  # Jitter to prevent thundering herd
  jitter_min: float = 0.1  # Minimum jitter factor (10%)
  jitter_max: float = 0.3  # Maximum jitter factor (30%)

  # State query specific
  max_state_query_retries: int = 3  # Maximum retries for E122 errors
  state_query_retry_delay: float = 2.0  # Delay between state query retries

  # Message sending
  max_message_retries: int = 3  # Maximum retries for message sending
  retry_backoff_base: float = 2.0  # Base for message retry backoff
  max_retry_delay: float = 60.0  # Maximum retry delay

  @classmethod
  def from_env(cls) -> 'RetryConfig':
    """Create configuration from environment variables."""
    return cls(
        max_retries=int(os.getenv('FREECIV_MAX_RETRIES', '3')),
        backoff_base=float(os.getenv('FREECIV_BACKOFF_BASE', '1.5')),
        max_backoff_delay=float(os.getenv('FREECIV_MAX_BACKOFF', '30.0')),
    )


@dataclass
class ConnectionConfig:
  """WebSocket connection configuration.

  Timeouts and connection management settings.
  """
  # Connection timeouts
  connect_timeout: float = 10.0  # WebSocket connection timeout
  message_timeout: float = 30.0  # Message send/receive timeout
  ping_timeout: float = 10.0  # Ping/pong timeout
  close_timeout: float = 5.0  # Connection close timeout

  # Heartbeat/keepalive
  heartbeat_interval: float = 30.0  # Interval between heartbeat messages

  # Reconnection
  max_reconnect_attempts: int = 3  # Maximum reconnection attempts
  session_resumption_window: float = 60.0  # Session resumption window (seconds)

  # Game initialization
  game_start_wait_seconds: float = 12.0  # Wait for game initialization

  @classmethod
  def from_env(cls) -> 'ConnectionConfig':
    """Create configuration from environment variables."""
    return cls(
        connect_timeout=float(os.getenv('FREECIV_CONNECT_TIMEOUT', '10.0')),
        message_timeout=float(os.getenv('FREECIV_MESSAGE_TIMEOUT', '30.0')),
        heartbeat_interval=float(os.getenv('FREECIV_HEARTBEAT_INTERVAL', '30.0')),
        max_reconnect_attempts=int(os.getenv('FREECIV_MAX_RECONNECT', '3')),
    )


@dataclass
class CacheConfig:
  """Caching configuration for performance optimization.

  Controls state caching, LRU cache sizes, and TTL values.
  """
  # State caching
  state_cache_ttl: float = 5.0  # Time-to-live for cached states
  max_cache_entries: int = 10  # LRU cache size

  # Parser caches
  action_cache_size: int = 1000  # Action string cache size
  string_cache_size: int = 1000  # String parsing cache size
  similarity_cache_size: int = 100  # Similarity computation cache
  memory_cache_ttl: float = 120.0  # TTL for high-frequency caches

  # Background task management
  background_task_shutdown_timeout: float = 5.0  # Graceful shutdown timeout

  @classmethod
  def from_env(cls) -> 'CacheConfig':
    """Create configuration from environment variables."""
    return cls(
        state_cache_ttl=float(os.getenv('FREECIV_STATE_CACHE_TTL', '5.0')),
        max_cache_entries=int(os.getenv('FREECIV_MAX_CACHE_ENTRIES', '10')),
        action_cache_size=int(os.getenv('FREECIV_ACTION_CACHE_SIZE', '1000')),
    )


@dataclass
class ProxyClientConfig:
  """Root configuration aggregating all sub-configurations.

  This provides a single entry point for all FreeCiv proxy client
  configuration with support for environment variable overrides.
  """
  security: SecurityConfig = field(default_factory=SecurityConfig)
  circuit_breaker: CircuitBreakerConfig = field(
      default_factory=CircuitBreakerConfig
  )
  rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
  retry: RetryConfig = field(default_factory=RetryConfig)
  connection: ConnectionConfig = field(default_factory=ConnectionConfig)
  cache: CacheConfig = field(default_factory=CacheConfig)

  @classmethod
  def from_env(cls) -> 'ProxyClientConfig':
    """Create complete configuration from environment variables.

    Returns:
        ProxyClientConfig with all sub-configs populated from environment.
    """
    return cls(
        security=SecurityConfig.from_env(),
        circuit_breaker=CircuitBreakerConfig.from_env(),
        rate_limit=RateLimitConfig.from_env(),
        retry=RetryConfig.from_env(),
        connection=ConnectionConfig.from_env(),
        cache=CacheConfig.from_env(),
    )

  def validate(self) -> None:
    """Validate configuration values are sensible.

    Raises:
        ValueError: If configuration contains invalid values.
    """
    # Security validations
    if self.security.max_json_size <= 0:
      raise ValueError("max_json_size must be positive")
    if self.security.max_json_depth <= 0:
      raise ValueError("max_json_depth must be positive")

    # Rate limit validations
    if self.rate_limit.requests_per_minute <= 0:
      raise ValueError("requests_per_minute must be positive")
    if self.rate_limit.burst_size <= 0:
      raise ValueError("burst_size must be positive")

    # Retry validations
    if self.retry.max_retries < 0:
      raise ValueError("max_retries cannot be negative")
    if self.retry.backoff_base <= 1.0:
      raise ValueError("backoff_base must be > 1.0 for exponential growth")
    if self.retry.jitter_min < 0 or self.retry.jitter_max > 1:
      raise ValueError("jitter values must be in range [0, 1]")
    if self.retry.jitter_min > self.retry.jitter_max:
      raise ValueError("jitter_min must be <= jitter_max")

    # Connection validations
    if self.connection.connect_timeout <= 0:
      raise ValueError("connect_timeout must be positive")
    if self.connection.max_reconnect_attempts < 0:
      raise ValueError("max_reconnect_attempts cannot be negative")

    # Cache validations
    if self.cache.state_cache_ttl <= 0:
      raise ValueError("state_cache_ttl must be positive")
    if self.cache.max_cache_entries <= 0:
      raise ValueError("max_cache_entries must be positive")


# Default configuration instance
DEFAULT_CONFIG = ProxyClientConfig()
