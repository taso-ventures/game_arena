"""FreeCiv Proxy Client for WebSocket communication with FreeCiv3D server."""

import asyncio
import hashlib
import json
import logging
import random
import copy
import re
import time
import uuid
from collections import OrderedDict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypedDict, Union

import websockets
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import ConnectionClosed, WebSocketException

from game_arena.harness.freeciv_state import FreeCivAction

logger = logging.getLogger(__name__)


# Type definitions for better type safety
class GameStateDict(TypedDict, total=False):
  """Type definition for game state dictionary."""
  turn: int
  playerID: int
  players: List[Dict[str, Any]]
  map: Dict[str, Any]
  phase: str
  game: Dict[str, Any]


class ActionRequestDict(TypedDict):
  """Type definition for action request dictionary."""
  type: str
  agent_id: str
  data: Dict[str, Any]


class StateRequestDict(TypedDict):
  """Type definition for state request dictionary."""
  type: str
  format: str
  agent_id: str


class AuthMessageDict(TypedDict):
  """Type definition for authentication message."""
  type: str
  agent_id: str
  timestamp: float
  data: Dict[str, Any]


class CircuitBreakerStatsDict(TypedDict):
  """Type definition for circuit breaker statistics."""
  state: str
  failure_count: int
  success_count: int
  last_failure_time: float
  last_state_change: float
  time_since_last_failure: float


class RateLimiterStatsDict(TypedDict):
  """Type definition for rate limiter statistics."""
  total_buckets: int
  requests_per_minute: int
  burst_size: int
  buckets: Dict[str, Dict[str, float]]

# Security constants - these should be configurable in production
MAX_JSON_SIZE = 5_000_000  # 5MB limit for JSON messages
MAX_JSON_DEPTH = 100  # Maximum nesting depth to prevent DoS
MAX_WEBSOCKET_SIZE = 5 * 10**6  # 5MB limit for WebSocket messages
MAX_WEBSOCKET_QUEUE = 32  # Maximum queued messages
WEBSOCKET_CLOSE_TIMEOUT = 5.0  # Timeout for WebSocket close operations

# Input validation constants
MAX_ACTION_ID = 999999  # Maximum valid action ID
MIN_ACTION_ID = 0  # Minimum valid action ID
MAX_PLAYER_ID = 16  # Maximum number of players in FreeCiv
MIN_PLAYER_ID = 0  # Minimum player ID
MAX_OBSERVATION_SIZE = 5_000_000  # 5MB limit for observation data
MAX_STRING_LENGTH = 10000  # Maximum length for string fields
MAX_ARRAY_LENGTH = 10000  # Maximum length for arrays

# Rate limiting and circuit breaker constants - configurable
DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 60.0
DEFAULT_CIRCUIT_BREAKER_SUCCESS_THRESHOLD = 3

# Rate limiting constants
DEFAULT_RATE_LIMIT_REQUESTS_PER_MINUTE = 60  # 60 requests per minute per player
DEFAULT_RATE_LIMIT_BURST_SIZE = 10  # Allow burst of 10 requests
DEFAULT_EXPONENTIAL_BACKOFF_BASE = 1.5  # Base for exponential backoff
DEFAULT_MAX_BACKOFF_DELAY = 30.0  # Maximum backoff delay in seconds

# Message size rate limiting - protect against bandwidth DoS
MAX_MESSAGES_PER_SECOND = 10  # Maximum messages per second per connection
MAX_BYTES_PER_MINUTE = 50_000_000  # 50MB per minute per connection
MESSAGE_RATE_WINDOW_SECONDS = 60.0  # Rolling window for message size tracking

# Cache key validation pattern
CACHE_KEY_PATTERN = re.compile(r'^[a-zA-Z0-9_.-]+$')


class PacketID(Enum):
  """FreeCiv packet type identifiers."""

  UNIT_ORDERS = 31
  CITY_CHANGE_PRODUCTION = 85
  PACKET_CONN_PING = 88
  PACKET_CONN_PONG = 89
  GENERIC = 0


# Configuration constants - these should be configurable per deployment
DEFAULT_HEARTBEAT_INTERVAL = 30.0
DEFAULT_STATE_CACHE_TTL = 5.0
DEFAULT_MAX_CACHE_ENTRIES = 10  # LRU cache size - consider adaptive sizing
DEFAULT_MAX_RECONNECT_ATTEMPTS = 3
BACKGROUND_TASK_SHUTDOWN_TIMEOUT = 5.0

# Retry configuration for state queries and actions
MAX_STATE_QUERY_RETRIES = 3  # Maximum retries for E122 errors
DEFAULT_RETRY_DELAY_SECONDS = 2.0  # Delay between retries
GAME_START_WAIT_SECONDS = 12  # Wait for nation selection + async registration + game start

# Parser security constants
MAX_TARGET_STRING_LENGTH = 200  # Maximum length for target strings in Python repr parsing

# Cache size constants - consider making adaptive based on memory
DEFAULT_ACTION_CACHE_SIZE = 1000
DEFAULT_STRING_CACHE_SIZE = 1000
DEFAULT_SIMILARITY_CACHE_SIZE = 100
DEFAULT_MEMORY_CACHE_TTL = 120.0  # 2 minutes for high-frequency games

# WebSocket timeout configurations
DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_MESSAGE_TIMEOUT = 30.0
DEFAULT_PING_TIMEOUT = 10.0
DEFAULT_CLOSE_TIMEOUT = 5.0
MAX_MESSAGE_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0
MAX_RETRY_DELAY = 60.0


def _count_json_depth(obj: Any, depth: int = 0) -> int:
  """Count the maximum nesting depth of a JSON object."""
  if depth > MAX_JSON_DEPTH:
      raise ValueError(f"JSON depth exceeds maximum of {MAX_JSON_DEPTH}")

  if isinstance(obj, dict):
      return max(
          (_count_json_depth(v, depth + 1) for v in obj.values()),
          default=depth
      )
  elif isinstance(obj, list):
      return max(
          (_count_json_depth(item, depth + 1) for item in obj),
          default=depth
      )
  return depth


def validate_action_id(action_id: Any) -> int:
  """Validate action ID is within acceptable range.

  Args:
      action_id: Action ID to validate

  Returns:
      Validated action ID as integer

  Raises:
      ValueError: If action ID is invalid or out of range
  """
  if not isinstance(action_id, (int, float)):
    try:
      action_id = int(action_id)
    except (ValueError, TypeError):
      raise ValueError(f"Action ID must be numeric, got {type(action_id)}")

  action_id = int(action_id)
  if action_id < MIN_ACTION_ID or action_id > MAX_ACTION_ID:
    raise ValueError(f"Action ID {action_id} out of valid range [{MIN_ACTION_ID}, {MAX_ACTION_ID}]")

  return action_id


def validate_player_id(player_id: Any) -> int:
  """Validate player ID is within acceptable range.

  Args:
      player_id: Player ID to validate

  Returns:
      Validated player ID as integer

  Raises:
      ValueError: If player ID is invalid or out of range
  """
  if not isinstance(player_id, (int, float)):
    try:
      player_id = int(player_id)
    except (ValueError, TypeError):
      raise ValueError(f"Player ID must be numeric, got {type(player_id)}")

  player_id = int(player_id)
  if player_id < MIN_PLAYER_ID or player_id > MAX_PLAYER_ID:
    raise ValueError(f"Player ID {player_id} out of valid range [{MIN_PLAYER_ID}, {MAX_PLAYER_ID}]")

  return player_id


def validate_string_length(text: str, field_name: str = "string") -> str:
  """Validate string length is within acceptable limits.

  Args:
      text: String to validate
      field_name: Name of field for error messages

  Returns:
      Validated string

  Raises:
      ValueError: If string is too long
  """
  if not isinstance(text, str):
    raise ValueError(f"{field_name} must be a string, got {type(text)}")

  if len(text) > MAX_STRING_LENGTH:
    raise ValueError(f"{field_name} length {len(text)} exceeds maximum of {MAX_STRING_LENGTH}")

  return text


def validate_array_length(arr: List[Any], field_name: str = "array") -> List[Any]:
  """Validate array length is within acceptable limits.

  Args:
      arr: Array to validate
      field_name: Name of field for error messages

  Returns:
      Validated array

  Raises:
      ValueError: If array is too long
  """
  if not isinstance(arr, list):
    raise ValueError(f"{field_name} must be a list, got {type(arr)}")

  if len(arr) > MAX_ARRAY_LENGTH:
    raise ValueError(f"{field_name} length {len(arr)} exceeds maximum of {MAX_ARRAY_LENGTH}")

  return arr


def validate_observation_size(observation: Dict[str, Any]) -> Dict[str, Any]:
  """Validate observation data size to prevent DoS.

  Args:
      observation: Observation dictionary to validate

  Returns:
      Validated observation

  Raises:
      ValueError: If observation is too large
  """
  if not isinstance(observation, dict):
    raise ValueError(f"Observation must be a dictionary, got {type(observation)}")

  # Estimate size by serializing to JSON
  try:
    json_str = json.dumps(observation)
    size_bytes = len(json_str.encode('utf-8'))

    if size_bytes > MAX_OBSERVATION_SIZE:
      raise ValueError(f"Observation size {size_bytes} bytes exceeds maximum of {MAX_OBSERVATION_SIZE}")

  except (TypeError, ValueError) as e:
    raise ValueError(f"Failed to serialize observation: {e}")

  return observation


def safe_json_loads(json_string: str) -> Dict[str, Any]:
  """Safely parse JSON with size and depth validation.

  Args:
      json_string: JSON string to parse

  Returns:
      Parsed JSON object

  Raises:
      ValueError: If JSON is invalid, too large, or too deeply nested
      json.JSONDecodeError: If JSON syntax is invalid
  """
  if not isinstance(json_string, str):
      raise ValueError("Input must be a string")

  # Check size limit
  if len(json_string.encode('utf-8')) > MAX_JSON_SIZE:
      raise ValueError(f"JSON size {len(json_string)} exceeds maximum of {MAX_JSON_SIZE}")

  # Parse JSON
  try:
      data = json.loads(json_string)
  except json.JSONDecodeError as e:
      logger.warning(f"Invalid JSON received: {e}")
      raise

  # Validate depth
  try:
      _count_json_depth(data)
  except ValueError as e:
      logger.warning(f"JSON depth validation failed: {e}")
      raise

  # Basic schema validation - ensure it's a dict with reasonable structure
  if not isinstance(data, dict):
      raise ValueError("JSON must be an object at root level")

  return data


def create_secure_cache_key(prefix: str, identifier: str) -> str:
  """Create a secure cache key with validation.

  Args:
      prefix: Cache key prefix
      identifier: Unique identifier for the cached item

  Returns:
      Secure cache key

  Raises:
      ValueError: If inputs contain invalid characters
  """
  # Validate inputs
  if not isinstance(prefix, str) or not isinstance(identifier, str):
      raise ValueError("Prefix and identifier must be strings")

  # Sanitize inputs - only allow alphanumeric and safe characters
  clean_prefix = re.sub(r'[^a-zA-Z0-9_]', '', prefix)
  clean_identifier = re.sub(r'[^a-zA-Z0-9_]', '', identifier)

  if not clean_prefix or not clean_identifier:
      raise ValueError("Invalid characters in cache key components")

  # Create hash of original identifier to prevent collision attacks
  identifier_hash = hashlib.sha256(identifier.encode('utf-8')).hexdigest()[:16]

  return f"{clean_prefix}_{clean_identifier}_{identifier_hash}"


class ConnectionState(Enum):
  """Connection state enumeration."""

  DISCONNECTED = "disconnected"
  CONNECTING = "connecting"
  CONNECTED = "connected"
  RECONNECTING = "reconnecting"


class CircuitBreakerState(Enum):
  """Circuit breaker state enumeration."""

  CLOSED = "closed"     # Normal operation
  OPEN = "open"         # Failing, rejecting requests
  HALF_OPEN = "half_open"  # Testing if service recovered


class RateLimiter:
  """Token bucket rate limiter with per-player limits."""

  def __init__(
      self,
      requests_per_minute: int = DEFAULT_RATE_LIMIT_REQUESTS_PER_MINUTE,
      burst_size: int = DEFAULT_RATE_LIMIT_BURST_SIZE,
  ):
    """Initialize rate limiter.

    Args:
        requests_per_minute: Maximum requests per minute
        burst_size: Maximum burst size
    """
    self.requests_per_minute = requests_per_minute
    self.burst_size = burst_size
    self.refill_rate = requests_per_minute / 60.0  # tokens per second

    # Per-player token buckets: {player_id: {"tokens": float, "last_refill": float}}
    self.player_buckets: Dict[str, Dict[str, float]] = {}

  def _get_bucket_key(self, player_id: Optional[str] = None, agent_id: Optional[str] = None) -> str:
    """Get rate limiting bucket key for player/agent."""
    if player_id:
      return f"player_{player_id}"
    elif agent_id:
      return f"agent_{agent_id}"
    else:
      return "default"

  def _refill_bucket(self, bucket: Dict[str, float]) -> None:
    """Refill token bucket based on elapsed time."""
    current_time = time.time()
    last_refill = bucket.get("last_refill", current_time)
    elapsed = current_time - last_refill

    # Add tokens based on elapsed time
    tokens_to_add = elapsed * self.refill_rate
    bucket["tokens"] = min(self.burst_size, bucket.get("tokens", self.burst_size) + tokens_to_add)
    bucket["last_refill"] = current_time

  def can_proceed(self, player_id: Optional[str] = None, agent_id: Optional[str] = None) -> bool:
    """Check if request can proceed without hitting rate limit.

    Args:
        player_id: Player ID for rate limiting
        agent_id: Agent ID for rate limiting (fallback)

    Returns:
        True if request can proceed, False if rate limited
    """
    bucket_key = self._get_bucket_key(player_id, agent_id)

    # Initialize bucket if not exists
    if bucket_key not in self.player_buckets:
      self.player_buckets[bucket_key] = {
          "tokens": self.burst_size,
          "last_refill": time.time()
      }

    bucket = self.player_buckets[bucket_key]
    self._refill_bucket(bucket)

    # Check if we have tokens available
    if bucket["tokens"] >= 1.0:
      bucket["tokens"] -= 1.0
      return True

    return False

  def time_until_next_request(self, player_id: Optional[str] = None, agent_id: Optional[str] = None) -> float:
    """Get time until next request can be made.

    Args:
        player_id: Player ID for rate limiting
        agent_id: Agent ID for rate limiting (fallback)

    Returns:
        Time in seconds until next request can be made
    """
    bucket_key = self._get_bucket_key(player_id, agent_id)

    if bucket_key not in self.player_buckets:
      return 0.0

    bucket = self.player_buckets[bucket_key]
    self._refill_bucket(bucket)

    if bucket["tokens"] >= 1.0:
      return 0.0

    # Calculate time needed to refill one token
    return (1.0 - bucket["tokens"]) / self.refill_rate

  def get_stats(self) -> RateLimiterStatsDict:
    """Get rate limiter statistics.

    Returns:
        Dictionary with rate limiter stats
    """
    current_time = time.time()
    stats = {
        "total_buckets": len(self.player_buckets),
        "requests_per_minute": self.requests_per_minute,
        "burst_size": self.burst_size,
        "buckets": {}
    }

    for bucket_key, bucket in self.player_buckets.items():
      self._refill_bucket(bucket)
      stats["buckets"][bucket_key] = {
          "tokens": bucket["tokens"],
          "last_refill": bucket["last_refill"],
          "time_since_refill": current_time - bucket["last_refill"]
      }

    return stats


class MessageSizeRateLimiter:
  """Rate limiter for message size to prevent bandwidth DoS attacks.

  Tracks both message count and total bytes transferred in rolling windows.
  """

  def __init__(
      self,
      max_messages_per_second: int = MAX_MESSAGES_PER_SECOND,
      max_bytes_per_minute: int = MAX_BYTES_PER_MINUTE,
      window_seconds: float = MESSAGE_RATE_WINDOW_SECONDS,
  ):
    """Initialize message size rate limiter.

    Args:
        max_messages_per_second: Maximum messages per second
        max_bytes_per_minute: Maximum bytes per minute
        window_seconds: Rolling window size in seconds
    """
    self.max_messages_per_second = max_messages_per_second
    self.max_bytes_per_minute = max_bytes_per_minute
    self.window_seconds = window_seconds

    # Per-connection tracking: {connection_id: {"messages": deque, "bytes": deque}}
    # deque contains tuples of (timestamp, size_bytes)
    self.connection_tracking: Dict[str, Dict[str, Any]] = {}

  def check_and_record(
      self,
      connection_id: str,
      message_size_bytes: int,
  ) -> bool:
    """Check rate limits and record message if allowed.

    Args:
        connection_id: Unique connection identifier
        message_size_bytes: Size of message in bytes

    Returns:
        True if message is allowed, False if rate limit exceeded
    """
    from collections import deque

    current_time = time.time()

    # Initialize tracking for new connections
    if connection_id not in self.connection_tracking:
      self.connection_tracking[connection_id] = {
          "messages": deque(),
          "bytes": deque(),
      }

    tracking = self.connection_tracking[connection_id]

    # Clean old entries outside the rolling windows
    # Message count window: 1 second
    message_window_start = current_time - 1.0
    while tracking["messages"] and tracking["messages"][0][0] < message_window_start:
      tracking["messages"].popleft()

    # Bytes count window: window_seconds (default 60)
    bytes_window_start = current_time - self.window_seconds
    while tracking["bytes"] and tracking["bytes"][0][0] < bytes_window_start:
      tracking["bytes"].popleft()

    # Check message rate limit (messages per second)
    if len(tracking["messages"]) >= self.max_messages_per_second:
      logger.warning(
          "Message rate limit exceeded for %s: %d messages in 1 second (max %d)",
          connection_id,
          len(tracking["messages"]),
          self.max_messages_per_second,
      )
      return False

    # Check byte rate limit (bytes per minute)
    total_bytes = sum(size for _, size in tracking["bytes"])
    if total_bytes + message_size_bytes > self.max_bytes_per_minute:
      logger.warning(
          "Bandwidth rate limit exceeded for %s: %d + %d bytes in %d seconds (max %d)",
          connection_id,
          total_bytes,
          message_size_bytes,
          self.window_seconds,
          self.max_bytes_per_minute,
      )
      return False

    # Record message
    tracking["messages"].append((current_time, message_size_bytes))
    tracking["bytes"].append((current_time, message_size_bytes))

    return True

  def get_stats(self, connection_id: str) -> Dict[str, Any]:
    """Get rate limiter statistics for a connection.

    Args:
        connection_id: Connection identifier

    Returns:
        Dictionary with current usage statistics
    """
    if connection_id not in self.connection_tracking:
      return {
          "messages_last_second": 0,
          "bytes_last_minute": 0,
          "message_limit": self.max_messages_per_second,
          "byte_limit": self.max_bytes_per_minute,
      }

    tracking = self.connection_tracking[connection_id]
    current_time = time.time()

    # Count recent messages
    message_count = sum(
        1 for ts, _ in tracking["messages"] if ts > current_time - 1.0
    )

    # Count recent bytes
    byte_count = sum(
        size for ts, size in tracking["bytes"] if ts > current_time - self.window_seconds
    )

    return {
        "messages_last_second": message_count,
        "bytes_last_minute": byte_count,
        "message_limit": self.max_messages_per_second,
        "byte_limit": self.max_bytes_per_minute,
        "message_utilization": message_count / self.max_messages_per_second,
        "byte_utilization": byte_count / self.max_bytes_per_minute,
    }


class CircuitBreaker:
  """Circuit breaker for API failure handling."""

  def __init__(
      self,
      failure_threshold: int = DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
      recovery_timeout: float = DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
      success_threshold: int = DEFAULT_CIRCUIT_BREAKER_SUCCESS_THRESHOLD,
  ):
    """Initialize circuit breaker.

    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time to wait before attempting recovery
        success_threshold: Number of successes needed to close circuit
    """
    self.failure_threshold = failure_threshold
    self.recovery_timeout = recovery_timeout
    self.success_threshold = success_threshold

    self.state = CircuitBreakerState.CLOSED
    self.failure_count = 0
    self.success_count = 0
    self.last_failure_time = 0
    self.last_state_change = time.time()

  def can_execute(self) -> bool:
    """Check if operation can be executed.

    Returns:
        True if operation should be allowed, False otherwise
    """
    current_time = time.time()

    if self.state == CircuitBreakerState.CLOSED:
      return True
    elif self.state == CircuitBreakerState.OPEN:
      # Check if recovery timeout has passed
      if current_time - self.last_failure_time >= self.recovery_timeout:
        self.state = CircuitBreakerState.HALF_OPEN
        self.success_count = 0
        logger.info("Circuit breaker transitioning to HALF_OPEN")
        return True
      return False
    elif self.state == CircuitBreakerState.HALF_OPEN:
      return True

    return False

  def record_success(self) -> None:
    """Record a successful operation."""
    if self.state == CircuitBreakerState.HALF_OPEN:
      self.success_count += 1
      if self.success_count >= self.success_threshold:
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_state_change = time.time()
        logger.info("Circuit breaker CLOSED - service recovered")
    elif self.state == CircuitBreakerState.CLOSED:
      # Reset failure count on success
      self.failure_count = 0

  def record_failure(self) -> None:
    """Record a failed operation."""
    self.failure_count += 1
    self.last_failure_time = time.time()

    if self.state == CircuitBreakerState.CLOSED:
      if self.failure_count >= self.failure_threshold:
        self.state = CircuitBreakerState.OPEN
        self.last_state_change = time.time()
        logger.warning(f"Circuit breaker OPEN - {self.failure_count} failures")
    elif self.state == CircuitBreakerState.HALF_OPEN:
      # Single failure in half-open state reopens circuit
      self.state = CircuitBreakerState.OPEN
      self.success_count = 0
      self.last_state_change = time.time()
      logger.warning("Circuit breaker reopened after failure in HALF_OPEN state")

  def get_state_info(self) -> CircuitBreakerStatsDict:
    """Get current circuit breaker state information.

    Returns:
        Dictionary with state information
    """
    return {
        "state": self.state.value,
        "failure_count": self.failure_count,
        "success_count": self.success_count,
        "last_failure_time": self.last_failure_time,
        "last_state_change": self.last_state_change,
        "time_since_last_failure": time.time() - self.last_failure_time,
    }


class FreeCivProxyClient:
  """WebSocket client for communicating with FreeCiv3D proxy server."""

  def __init__(
      self,
      host: str = "localhost",
      port: int = 8002,
      agent_id: Optional[str] = None,
      game_id: str = "default",
      api_token: Optional[str] = None,
      endpoint: str = "/llmsocket",
      heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
      state_cache_ttl: float = DEFAULT_STATE_CACHE_TTL,
      max_cache_entries: int = DEFAULT_MAX_CACHE_ENTRIES,
      nation: Optional[str] = None,
      leader_name: Optional[str] = None,
  ):
      """Initialize FreeCiv proxy client.

      Args:
          host: FreeCiv3D server host
          port: FreeCiv3D server port
          agent_id: Unique agent identifier
          game_id: Game session identifier
          api_token: API token for authentication with FreeCiv3D LLM gateway
          endpoint: WebSocket endpoint path (default: /llmsocket for FreeCiv3D LLM gateway)
          heartbeat_interval: Heartbeat interval in seconds
          state_cache_ttl: State cache TTL in seconds
          max_cache_entries: Maximum number of cache entries
          nation: Nation preference (e.g., "Americans", "Romans", or "random")
          leader_name: Leader name for the player
      """
      self.host = host
      self.port = port
      self.agent_id = agent_id or f"agent_{uuid.uuid4().hex[:8]}"
      self.game_id = game_id
      self.api_token = api_token
      self.endpoint = endpoint
      self.heartbeat_interval = heartbeat_interval
      self.state_cache_ttl = state_cache_ttl
      self.max_cache_entries = max_cache_entries
      self.nation = nation
      self.leader_name = leader_name or self.agent_id
      self.civserver_port = None  # Will be set from auth_success response

      # Connection management - construct WebSocket URL for LLM Gateway
      ws_url = f"ws://{host}:{port}/ws/agent/{self.agent_id}"
      self.connection_manager = ConnectionManager(
          ws_url=ws_url,
          agent_id=self.agent_id,
          heartbeat_interval=heartbeat_interval,
          max_reconnect_attempts=DEFAULT_MAX_RECONNECT_ATTEMPTS,
          connect_timeout=DEFAULT_CONNECT_TIMEOUT,
          message_timeout=DEFAULT_MESSAGE_TIMEOUT,
          ping_timeout=DEFAULT_PING_TIMEOUT,
          close_timeout=DEFAULT_CLOSE_TIMEOUT,
      )

      # Message handling
      self.message_handler = MessageHandler(client=self)
      self.message_queue = MessageQueue()
      self.protocol_translator = ProtocolTranslator()

      # State management
      self.player_id: Optional[int] = None
      self.state_cache: OrderedDict[str, Any] = OrderedDict()
      self._state_cache_lock: asyncio.Lock = asyncio.Lock()  # Protect concurrent cache access
      self.last_state_update = 0
      self.last_error: Optional[Dict[str, Any]] = None  # Store last error for retry logic
      self.last_action_error: Optional[Dict[str, Any]] = None  # Store last action rejection for debugging
      self.game_ready: bool = False  # Flag set when server sends game_ready message
      self.game_ready_event: asyncio.Event = asyncio.Event()  # Event for async waiting on game initialization

      # Background tasks
      self._heartbeat_task: Optional[asyncio.Task] = None
      self._message_processor_task: Optional[asyncio.Task] = None

      # Circuit breaker for API failure handling
      self.circuit_breaker = CircuitBreaker(
          failure_threshold=5,
          recovery_timeout=60.0,
          success_threshold=3,
      )

      # Message size rate limiter for bandwidth DoS protection
      self.message_size_limiter = MessageSizeRateLimiter(
          max_messages_per_second=MAX_MESSAGES_PER_SECOND,
          max_bytes_per_minute=MAX_BYTES_PER_MINUTE,
          window_seconds=MESSAGE_RATE_WINDOW_SECONDS,
      )

      # Rate limiter for API requests
      self.rate_limiter = RateLimiter(
          requests_per_minute=DEFAULT_RATE_LIMIT_REQUESTS_PER_MINUTE,
          burst_size=DEFAULT_RATE_LIMIT_BURST_SIZE,
      )

  async def connect(self) -> bool:
      """Connect to FreeCiv3D server.

      Returns:
          True if connection successful, False otherwise
      """
      try:
          # Establish WebSocket connection
          success = await self.connection_manager.connect()
          if not success:
              return False

          # Wait for welcome message
          welcome_response = await self.connection_manager.receive_message()
          if welcome_response:
              try:
                  welcome_data = safe_json_loads(welcome_response)
                  logger.debug(f"Received welcome: {welcome_data}")
              except (ValueError, json.JSONDecodeError) as e:
                  logger.error(f"Invalid welcome message: {e}")
                  return False

          # Send authentication message with nation preferences
          auth_data = {
              "api_token": self.api_token or "test-token-fc3d-001",
              "model": "gpt-4",
              "game_id": self.game_id,
              "capabilities": ["move", "build", "research"]
          }

          # Add nation preferences if provided
          if self.nation:
              auth_data["nation"] = self.nation
          if self.leader_name:
              auth_data["leader_name"] = self.leader_name

          auth_message = {
              "type": "llm_connect",
              "agent_id": self.agent_id,
              "timestamp": time.time(),
              "data": auth_data
          }

          # Send authentication message (no logging to avoid credential leakage)
          logger.debug(f"Sending authentication for agent {self.agent_id}")
          await self.connection_manager.send_message(json.dumps(auth_message))

          # Wait for authentication response
          auth_response = await self.connection_manager.receive_message()
          if auth_response:
              try:
                  auth_data = safe_json_loads(auth_response)
              except (ValueError, json.JSONDecodeError) as e:
                  logger.error(f"Invalid authentication response: {e}")
                  return False

              # Check nested auth success in data field
              data_section = auth_data.get("data", {})
              if (auth_data.get("type") == "llm_connect" and
                  (data_section.get("type") == "auth_success" or data_section.get("success") == True)):
                  self.player_id = data_section.get("player_id")
                  self.civserver_port = data_section.get("civserver_port")  # Store civserver port for spectator URLs
                  logger.info(
                      f"Successfully authenticated as player {self.player_id} on civserver port {self.civserver_port}"
                  )

                  # Server now sends only ONE auth_success message with player_id already assigned
                  # The server waits for PACKET_CONN_INFO before sending auth_success
                  # So player_id should always be set
                  if self.player_id is None:
                      logger.warning("Received auth_success but player_id is None - server may not have assigned player yet")

                  # Start background tasks
                  await self._start_background_tasks()
                  return True
              else:
                  # Redact sensitive data before logging authentication failure
                  auth_data_safe = copy.deepcopy(auth_data)
                  if "data" in auth_data_safe and isinstance(auth_data_safe["data"], dict):
                      if "api_token" in auth_data_safe["data"]:
                          auth_data_safe["data"]["api_token"] = "***REDACTED***"
                  logger.error(f"Authentication failed: {auth_data_safe}")

          return False

      except Exception as e:
          logger.error(f"Failed to connect: {e}")
          return False

  async def disconnect(self) -> None:
      """Disconnect from FreeCiv3D server."""
      # Stop background tasks
      await self._stop_background_tasks()

      # Disconnect WebSocket
      await self.connection_manager.disconnect()

      # Clear state
      self.player_id = None

      # Protect cache clear with async lock
      async with self._state_cache_lock:
          self.state_cache.clear()

  async def wait_for_game_ready(self, timeout: float = 30.0) -> bool:
      """Wait for game_ready signal from server with timeout.

      This method blocks until the server broadcasts the game_ready message,
      indicating that the game is fully initialized with units, cities, and
      nations assigned.

      Args:
          timeout: Maximum time to wait in seconds (default 30s)

      Returns:
          True if game_ready received, False if timeout

      Example:
          if await proxy.wait_for_game_ready(timeout=20.0):
              print("Game ready!")
              state = await proxy.get_state()
          else:
              print("Timeout waiting for game to start")
      """
      try:
          await asyncio.wait_for(self.game_ready_event.wait(), timeout=timeout)
          logger.info(f"✅ Game ready signal received within {timeout}s")
          return True
      except asyncio.TimeoutError:
          logger.warning(f"⚠️ Timeout waiting for game_ready signal after {timeout}s")
          return False

  async def get_state(
      self,
      format_type: str = "llm_optimized",
      max_retries: int = 3,
      retry_delay: float = 2.0
  ) -> GameStateDict:
      """Get current game state with retry logic for E122 errors.

      Args:
          format_type: State format ("llm_optimized", "minimal", etc.)
          max_retries: Maximum retry attempts for E122 errors (player not ready)
          retry_delay: Delay between retries in seconds

      Returns:
          Current game state dictionary

      Raises:
          RuntimeError: If not connected to FreeCiv server or failed to get state
          ConnectionClosed: If WebSocket connection is lost during request
          json.JSONDecodeError: If server response cannot be parsed
          asyncio.TimeoutError: If state request times out
      """
      if self.connection_manager.state != ConnectionState.CONNECTED:
          raise RuntimeError("Not connected to FreeCiv server")

      # Check circuit breaker
      if not self.circuit_breaker.can_execute():
          raise RuntimeError(f"Circuit breaker is {self.circuit_breaker.state.value} - rejecting get_state request")

      # Check rate limiter and apply exponential backoff if needed
      await self._enforce_rate_limit()

      # Check cache first
      try:
          cache_key = create_secure_cache_key("state", format_type)
      except ValueError as e:
          logger.error(f"Invalid format_type for cache key: {e}")
          raise ValueError(f"Invalid format_type: {format_type}")
      current_time = time.time()

      # Protect cache read with async lock to prevent race conditions
      async with self._state_cache_lock:
          if (
              cache_key in self.state_cache
              and current_time - self.state_cache[cache_key].get("_timestamp", 0)
              < self.state_cache_ttl
          ):
              # Move to end (most recently used) for LRU
              cached_item = self.state_cache.pop(cache_key)
              self.state_cache[cache_key] = cached_item
              return {
                  k: v
                  for k, v in cached_item.items()
                  if k != "_timestamp"
              }

      # Request fresh state with retry logic for E122 errors
      for attempt in range(max_retries):
          state_request = {
              "type": "state_query",
              "format": format_type,
              "agent_id": self.agent_id,
          }

          try:
              message_str = json.dumps(state_request)
              message_size = len(message_str.encode('utf-8'))

              # Check message size rate limit before sending
              if not self.message_size_limiter.check_and_record(self.agent_id, message_size):
                  raise RuntimeError(
                      f"Message rate limit exceeded. "
                      f"Stats: {self.message_size_limiter.get_stats(self.agent_id)}"
                  )

              logger.debug(f"Sending STATE_QUERY (attempt {attempt + 1}/{max_retries})")
              await self.connection_manager.send_message(message_str)

              # Wait for either state_update or error response
              response = await self._wait_for_response(["state_update", "error"])
              logger.debug(f"Received response type: {response.get('type') if response else 'None'}")

              # Check if we received an error
              if response and response.get("type") == "error":
                  error_data = response.get("data", {})
                  error_code = error_data.get("code", "UNKNOWN")
                  error_message = error_data.get("message", "Unknown error")

                  # E122 means player not ready yet - retry if attempts remain
                  if error_code == "E122" and attempt < max_retries - 1:
                      logger.warning(
                          f"⏳ Player not ready (E122), retrying in {retry_delay}s "
                          f"(attempt {attempt + 1}/{max_retries})"
                      )
                      await asyncio.sleep(retry_delay)
                      continue  # Retry
                  else:
                      # Other errors or max retries reached - fail
                      self.circuit_breaker.record_failure()
                      raise RuntimeError(
                          f"Failed to get game state - [{error_code}]: {error_message}"
                      )

              # Check if we received a valid state_update
              if response and response.get("type") == "state_update":
                  logger.debug("STATE_UPDATE received successfully")
                  # Record success with circuit breaker
                  self.circuit_breaker.record_success()

                  # Protect cache write and eviction with async lock
                  async with self._state_cache_lock:
                      # Evict old cache entries if limit exceeded using LRU
                      if len(self.state_cache) >= self.max_cache_entries:
                          # Remove least recently used entry (first item in OrderedDict)
                          oldest_key, _ = self.state_cache.popitem(last=False)
                          logger.debug(f"Evicted LRU cache entry: {oldest_key}")

                      # Cache the response with timestamp
                      self.state_cache[cache_key] = {**response, "_timestamp": current_time}

                  self.last_state_update = current_time
                  return response

              # No valid response - retry if attempts remain
              if attempt < max_retries - 1:
                  logger.warning(f"⚠️ No valid response, retrying in {retry_delay}s")
                  await asyncio.sleep(retry_delay)
                  continue
              else:
                  # Max retries exhausted
                  self.circuit_breaker.record_failure()
                  raise RuntimeError("Failed to get game state - no response")

          except RuntimeError:
              # Re-raise RuntimeError as-is (includes our error messages)
              raise
          except Exception as e:
              # Other exceptions - retry if attempts remain
              if attempt < max_retries - 1:
                  logger.warning(f"⚠️ State query error: {e}, retrying in {retry_delay}s")
                  await asyncio.sleep(retry_delay)
                  continue
              else:
                  # Max retries exhausted
                  self.circuit_breaker.record_failure()
                  raise

  async def send_action(self, action: FreeCivAction) -> Dict[str, Any]:
      """Send action to FreeCiv server.

      Args:
          action: FreeCivAction to send

      Returns:
          Action result dictionary

      Raises:
          RuntimeError: If not connected to FreeCiv server or failed to send action
          ConnectionClosed: If WebSocket connection is lost during request
          json.JSONDecodeError: If server response cannot be parsed
          asyncio.TimeoutError: If action request times out
          ValueError: If action format is invalid
      """
      if self.connection_manager.state != ConnectionState.CONNECTED:
          raise RuntimeError("Not connected to FreeCiv server")

      # Check circuit breaker
      if not self.circuit_breaker.can_execute():
          raise RuntimeError(f"Circuit breaker is {self.circuit_breaker.state.value} - rejecting send_action request")

      # Check rate limiter and apply exponential backoff if needed
      await self._enforce_rate_limit()

      # Convert action to packet format
      packet = self.protocol_translator.to_freeciv_packet(action)

      action_request = {
          "type": "action",
          "agent_id": self.agent_id,
          "data": packet,
      }

      # Send directly for now (message queue can be used for batching later)
      try:
          message_str = json.dumps(action_request)
          message_size = len(message_str.encode('utf-8'))

          # Check message size rate limit before sending
          if not self.message_size_limiter.check_and_record(self.agent_id, message_size):
              raise RuntimeError(
                  f"Message rate limit exceeded. "
                  f"Stats: {self.message_size_limiter.get_stats(self.agent_id)}"
              )

          await self.connection_manager.send_message(message_str)
          response = await self._wait_for_response(["action_result", "action_accepted", "action_rejected"])

          if response:
              # Record success with circuit breaker
              self.circuit_breaker.record_success()

              # Normalize response format for backward compatibility
              msg_type = response.get("type", "")
              if msg_type == "action_accepted":
                  # New format - normalize to old format
                  return {
                      "success": True,
                      "type": "action_accepted",
                      "data": response.get("data", response)
                  }
              elif msg_type == "action_rejected":
                  # New format - normalize to old format with error
                  error_data = response.get("data", {})
                  return {
                      "success": False,
                      "type": "action_rejected",
                      "error": error_data.get("error_message", "Action rejected"),
                      "error_code": error_data.get("error_code", "UNKNOWN"),
                      "data": error_data
                  }
              else:
                  # Old format or unknown - return as-is
                  return response

          # Record failure if no response received
          self.circuit_breaker.record_failure()
          raise RuntimeError("Failed to send action")

      except Exception as e:
          # Record failure with circuit breaker
          self.circuit_breaker.record_failure()
          raise

  async def _enforce_rate_limit(self) -> None:
      """Enforce rate limiting with exponential backoff.

      Raises:
          RuntimeError: If rate limit cannot be satisfied within reasonable time
      """
      player_key = str(self.player_id) if self.player_id else None
      agent_key = self.agent_id

      max_attempts = 5
      for attempt in range(max_attempts):
          if self.rate_limiter.can_proceed(player_id=player_key, agent_id=agent_key):
              return  # Rate limit satisfied

          # Calculate backoff delay
          base_delay = self.rate_limiter.time_until_next_request(
              player_id=player_key, agent_id=agent_key
          )

          # Apply exponential backoff with jitter
          backoff_multiplier = DEFAULT_EXPONENTIAL_BACKOFF_BASE ** attempt
          jitter = random.uniform(0.1, 0.3)  # 10-30% jitter
          delay = min(base_delay * backoff_multiplier * (1 + jitter), DEFAULT_MAX_BACKOFF_DELAY)

          logger.warning(
              f"Rate limited, waiting {delay:.2f}s (attempt {attempt + 1}/{max_attempts})"
          )
          await asyncio.sleep(delay)

      # If we've exhausted all attempts, raise an error
      raise RuntimeError(
          f"Rate limit exceeded after {max_attempts} attempts, "
          f"requests limited to {self.rate_limiter.requests_per_minute}/min"
      )

  async def _start_background_tasks(self) -> None:
      """Start background tasks for heartbeat and message processing."""
      self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
      self._message_processor_task = asyncio.create_task(
          self._message_processor_loop()
      )

  async def _stop_background_tasks(self) -> None:
      """Stop background tasks with graceful shutdown."""
      tasks = []

      # Cancel tasks
      if self._heartbeat_task and not self._heartbeat_task.done():
          self._heartbeat_task.cancel()
          tasks.append(self._heartbeat_task)

      if self._message_processor_task and not self._message_processor_task.done():
          self._message_processor_task.cancel()
          tasks.append(self._message_processor_task)

      # Wait for all tasks with timeout
      if tasks:
          try:
              done, pending = await asyncio.wait(
                  tasks,
                  timeout=BACKGROUND_TASK_SHUTDOWN_TIMEOUT,
                  return_when=asyncio.ALL_COMPLETED,
              )

              # Force cancel any remaining tasks and await them
              for task in pending:
                  task.cancel()
                  try:
                      await task
                  except asyncio.CancelledError:
                      pass  # Expected for cancelled tasks
                  except Exception as e:
                      logger.warning(f"Error in cancelled task {task.get_name()}: {e}")

              # Ensure all done tasks are properly awaited to handle any exceptions
              for task in done:
                  try:
                      await task
                  except asyncio.CancelledError:
                      pass  # Expected for cancelled tasks
                  except Exception as e:
                      logger.warning(f"Error in completed task {task.get_name()}: {e}")

          except Exception as e:
              logger.error(f"Error during background task shutdown: {e}")
          finally:
              # Ensure all task references are cleared
              self._heartbeat_task = None
              self._message_processor_task = None

  async def _heartbeat_loop(self) -> None:
      """Background heartbeat loop."""
      while self.connection_manager.state == ConnectionState.CONNECTED:
          try:
              ping_message = {
                  "type": "ping",
                  "timestamp": int(time.time()),
                  "echo": f"heartbeat_{uuid.uuid4().hex[:8]}",
              }
              await self.message_queue.enqueue(ping_message, priority=2)
              await asyncio.sleep(self.heartbeat_interval)
          except asyncio.CancelledError:
              break
          except Exception as e:
              logger.warning(f"Heartbeat error: {e}")

  async def _message_processor_loop(self) -> None:
      """Background message processing loop."""
      while self.connection_manager.state == ConnectionState.CONNECTED:
          try:
              await self.message_queue.process_messages(
                  self._send_message_to_server, max_messages=10
              )
              await asyncio.sleep(0.01)  # Small delay to prevent busy loop
          except asyncio.CancelledError:
              break
          except Exception as e:
              logger.warning(f"Message processing error: {e}")

  async def _send_message_to_server(self, message: Dict[str, Any]) -> None:
      """Send message to server via WebSocket.

      Args:
          message: Message to send
      """
      try:
          message_str = json.dumps(message)
          await self.connection_manager.send_message(message_str)
      except Exception as e:
          logger.error(f"Failed to send message: {e}")
          # Try to reconnect if connection is lost
          if self.connection_manager.state == ConnectionState.CONNECTED:
              await self.connection_manager.reconnect()

  async def _wait_for_response(
      self, expected_type: Union[str, List[str]], timeout: float = 30.0
  ) -> Optional[Dict[str, Any]]:
      """Wait for specific response type(s).

      Args:
          expected_type: Expected message type or list of types
          timeout: Timeout in seconds

      Returns:
          Response message or None if timeout/error

      Raises:
          RuntimeError: If connection is lost during wait
          json.JSONDecodeError: If response cannot be parsed as JSON
          ConnectionClosed: If WebSocket connection is closed unexpectedly
          WebSocketException: If other WebSocket errors occur
      """
      start_time = time.time()

      # Normalize expected_type to list for consistent handling
      expected_types = [expected_type] if isinstance(expected_type, str) else expected_type

      while time.time() - start_time < timeout:
          # Check if still connected
          if self.connection_manager.state != ConnectionState.CONNECTED:
              logger.warning("Connection lost while waiting for response")
              return None

          try:
              response = await asyncio.wait_for(
                  self.connection_manager.receive_message(), timeout=1.0
              )
              if response:
                  # Skip non-JSON messages (heartbeats, status updates, etc.)
                  # These are plain text messages that don't start with '{'
                  response_stripped = response.strip()
                  if not response_stripped or not response_stripped.startswith('{'):
                      # Silently skip non-JSON messages (common for server heartbeats)
                      logger.debug(f"Skipping non-JSON message: {response_stripped[:50]}")
                      continue

                  try:
                      data = safe_json_loads(response)
                      # Check if message type matches any expected type
                      if data.get("type") in expected_types:
                          return data
                      else:
                          # Handle other message types
                          await self.message_handler.handle_message(data)
                  except (ValueError, json.JSONDecodeError) as e:
                      # Only warn for messages that looked like JSON but failed to parse
                      logger.warning(f"Invalid JSON response (starts with '{{' but parse failed): {e}")
                      logger.debug(f"Problematic message: {response[:200]}")
                      continue
          except asyncio.TimeoutError:
              continue
          except Exception as e:
              logger.warning(f"Error waiting for response: {e}")
              # If we get repeated errors, break to prevent infinite loops
              break

      return None


class ConnectionManager:
  """Manages WebSocket connection with reconnection logic."""

  def __init__(
      self,
      ws_url: str,
      agent_id: str,
      heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
      max_reconnect_attempts: int = DEFAULT_MAX_RECONNECT_ATTEMPTS,
      connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
      message_timeout: float = DEFAULT_MESSAGE_TIMEOUT,
      ping_timeout: float = DEFAULT_PING_TIMEOUT,
      close_timeout: float = DEFAULT_CLOSE_TIMEOUT,
  ):
      """Initialize connection manager.

      Args:
          ws_url: WebSocket URL
          agent_id: Agent identifier
          heartbeat_interval: Heartbeat interval in seconds
          max_reconnect_attempts: Maximum reconnection attempts
          connect_timeout: Connection timeout in seconds
          message_timeout: Message timeout in seconds
          ping_timeout: Ping timeout in seconds
          close_timeout: Close timeout in seconds
      """
      self.ws_url = ws_url
      self.agent_id = agent_id
      self.heartbeat_interval = heartbeat_interval
      self.max_reconnect_attempts = max_reconnect_attempts
      self.connect_timeout = connect_timeout
      self.message_timeout = message_timeout
      self.ping_timeout = ping_timeout
      self.close_timeout = close_timeout

      self.websocket: Optional[WebSocketClientProtocol] = None
      self.state = ConnectionState.DISCONNECTED
      self.reconnect_attempts = 0
      self.last_error: Optional[Exception] = None

  async def connect(self) -> bool:
      """Establish WebSocket connection.

      Returns:
          True if successful, False otherwise
      """
      self.state = ConnectionState.CONNECTING

      try:
          # Use asyncio.wait_for to enforce connection timeout
          self.websocket = await asyncio.wait_for(
              websockets.connect(
                  self.ws_url,
                  ping_interval=self.heartbeat_interval,
                  ping_timeout=self.ping_timeout,
                  max_size=MAX_WEBSOCKET_SIZE,
                  max_queue=MAX_WEBSOCKET_QUEUE,
                  close_timeout=self.close_timeout
              ),
              timeout=self.connect_timeout
          )
          self.state = ConnectionState.CONNECTED
          self.reconnect_attempts = 0
          self.last_error = None
          logger.info(f"Connected to {self.ws_url}")
          return True

      except asyncio.TimeoutError:
          error_msg = f"Connection timeout after {self.connect_timeout}s"
          self.last_error = asyncio.TimeoutError(error_msg)
          logger.error(error_msg)
          self.state = ConnectionState.DISCONNECTED
          return False
      except Exception as e:
          self.last_error = e
          logger.error(f"Connection failed: {e}")
          self.state = ConnectionState.DISCONNECTED
          return False

  async def disconnect(self) -> None:
      """Close WebSocket connection."""
      if self.websocket:
          try:
              await asyncio.wait_for(
                  self.websocket.close(), timeout=self.close_timeout
              )
          except asyncio.TimeoutError:
              logger.warning(f"WebSocket close timed out after {self.close_timeout}s")
          except Exception as e:
              logger.warning(f"Error closing WebSocket: {e}")
          finally:
              self.websocket = None
      self.state = ConnectionState.DISCONNECTED

  async def reconnect(self) -> bool:
      """Attempt to reconnect with exponential backoff.

      Returns:
          True if reconnection successful, False otherwise
      """
      if self.reconnect_attempts >= self.max_reconnect_attempts:
          logger.error("Max reconnection attempts reached")
          # Reset for future manual reconnect attempts
          self.reconnect_attempts = 0
          return False

      self.state = ConnectionState.RECONNECTING
      self.reconnect_attempts += 1

      # Exponential backoff
      backoff_delay = self._calculate_backoff(self.reconnect_attempts)
      logger.info(
          f"Reconnecting in {backoff_delay}s (attempt {self.reconnect_attempts})"
      )
      await asyncio.sleep(backoff_delay)

      return await self.connect()

  def _calculate_backoff(self, attempt: int) -> float:
      """Calculate exponential backoff delay.

      Args:
          attempt: Attempt number

      Returns:
          Delay in seconds
      """
      return min(2**attempt, 60)  # Cap at 60 seconds

  async def send_message(self, message: str) -> None:
      """Send message via WebSocket with timeout and retry logic.

      Args:
          message: Message string to send

      Raises:
          RuntimeError: If not connected or max retries exceeded
          asyncio.TimeoutError: If message send times out
          ConnectionClosed: If connection is lost and reconnection fails
      """
      if not self.websocket or self.state != ConnectionState.CONNECTED:
          raise RuntimeError("Not connected")

      for attempt in range(MAX_MESSAGE_RETRIES):
          try:
              await asyncio.wait_for(
                  self.websocket.send(message), timeout=self.message_timeout
              )
              return  # Success, exit retry loop

          except asyncio.TimeoutError:
              error_msg = f"Message send timeout after {self.message_timeout}s (attempt {attempt + 1})"
              logger.warning(error_msg)
              if attempt == MAX_MESSAGE_RETRIES - 1:
                  raise asyncio.TimeoutError(error_msg)
              # Wait before retry
              await asyncio.sleep(RETRY_BACKOFF_BASE ** attempt)

          except (ConnectionClosed, WebSocketException) as e:
              logger.warning(f"Send failed, attempting reconnect (attempt {attempt + 1}): {e}")

              # Try to reconnect
              reconnect_success = await self.reconnect()
              if not reconnect_success:
                  if attempt == MAX_MESSAGE_RETRIES - 1:
                      raise RuntimeError(f"Failed to reconnect after {MAX_MESSAGE_RETRIES} attempts")
                  continue

              # If we reconnected successfully, try sending again
              continue

      raise RuntimeError(f"Failed to send message after {MAX_MESSAGE_RETRIES} attempts")

  async def receive_message(self) -> Optional[str]:
      """Receive message from WebSocket with timeout.

      Returns:
          Message string or None if error/timeout

      Raises:
          RuntimeError: If not connected
      """
      if not self.websocket or self.state != ConnectionState.CONNECTED:
          raise RuntimeError("Not connected")

      try:
          message = await asyncio.wait_for(
              self.websocket.recv(), timeout=self.message_timeout
          )
          return message
      except asyncio.TimeoutError:
          logger.warning(f"Message receive timeout after {self.message_timeout}s")
          return None
      except (ConnectionClosed, WebSocketException) as e:
          logger.warning(f"Receive failed, attempting reconnect: {e}")
          self.last_error = e
          # Don't await reconnect here to avoid blocking, let the caller decide
          self.state = ConnectionState.DISCONNECTED
          return None


class MessageHandler:
  """Handles incoming messages from FreeCiv server."""

  def __init__(self, client: Optional['FreeCivProxyClient'] = None):
      """Initialize message handler.

      Args:
          client: Reference to FreeCivProxyClient for storing state
      """
      self.client = client

  async def handle_message(self, message: Dict[str, Any]) -> None:
      """Route message to appropriate handler.

      Args:
          message: Parsed message dictionary
      """
      msg_type = message.get("type", "")

      if msg_type == "state_update":
          await self.handle_state_update(message)
      elif msg_type == "action_result":
          await self.handle_action_result(message)
      elif msg_type == "action_accepted":
          await self.handle_action_accepted(message)
      elif msg_type == "action_rejected":
          await self.handle_action_rejected(message)
      elif msg_type == "turn_notification":
          await self.handle_turn_notification(message)
      elif msg_type == "error":
          await self.handle_error(message)
      elif msg_type == "pong":
          await self.handle_pong(message)
      elif msg_type == "conn_ping":
          await self.handle_conn_ping(message)
      elif msg_type == "game_ready":
          await self.handle_game_ready(message)
      elif msg_type in ["welcome", "llm_connect"]:
          # Informational messages from server - log at debug level
          logger.debug(f"Server info message: {msg_type}")
      else:
          logger.warning(f"Unknown message type: {msg_type}")

  async def handle_state_update(self, message: Dict[str, Any]) -> None:
      """Handle state update from server.

      Args:
          message: State update message
      """
      logger.debug("Received state update")
      # State updates are handled by the client's response waiting logic

  async def handle_action_result(self, message: Dict[str, Any]) -> None:
      """Handle action result from server.

      Args:
          message: Action result message
      """
      success = message.get("success", False)
      logger.debug(f"Action result: success={success}")

  async def handle_action_accepted(self, message: Dict[str, Any]) -> None:
      """Handle action accepted confirmation from server.

      Args:
          message: Action accepted message with structure:
              {
                  "type": "action_accepted",
                  "action": {
                      "type": "tech_research",
                      "tech_name": "alphabet",
                      "player_id": 1
                  },
                  "timestamp": 1760891128.11
              }
              OR (with data wrapper):
              {
                  "type": "action_accepted",
                  "agent_id": "agent_player1_xxx",
                  "data": {...}
              }
      """
      # Handle both wrapped and unwrapped formats
      if "data" in message:
          action_data = message.get("data", {})
          action = action_data.get("action", {})
          timestamp = action_data.get("timestamp", "unknown")
      else:
          action = message.get("action", {})
          timestamp = message.get("timestamp", "unknown")

      logger.info(
          f"✅ Action accepted by FreeCiv server:\n"
          f"   Type: {action.get('type', 'unknown')}\n"
          f"   Details: {action}\n"
          f"   Timestamp: {timestamp}"
      )

  async def handle_action_rejected(self, message: Dict[str, Any]) -> None:
      """Handle action rejected error from server.

      Stores error details for debugging and analysis.

      Args:
          message: Action rejected message with structure:
              {
                  "type": "action_rejected",
                  "error_code": "E041",
                  "error_message": "Invalid technology...",
                  "action": {...},
                  "expected_format": {...},
                  "timestamp": 1760891128.11
              }
              OR (with data wrapper):
              {
                  "type": "action_rejected",
                  "agent_id": "agent_player1_xxx",
                  "data": {...}
              }
      """
      # Handle both wrapped and unwrapped formats
      if "data" in message:
          error_data = message.get("data", {})
      else:
          error_data = message

      error_code = error_data.get("error_code", "UNKNOWN")
      error_message = error_data.get("error_message", "Unknown error")
      action = error_data.get("action", {})
      expected_format = error_data.get("expected_format", {})

      logger.error(
          f"❌ Action rejected by FreeCiv server:\n"
          f"   Error Code: {error_code}\n"
          f"   Error Message: {error_message}\n"
          f"   Action: {action}\n"
          f"   Expected Format Example: {expected_format.get('json_example', 'N/A')}\n"
          f"   Notes: {expected_format.get('notes', 'N/A')}"
      )

      # Store last action error for debugging (similar to handle_error pattern)
      self.last_action_error = {
          "code": error_code,
          "message": error_message,
          "action": action,
          "expected_format": expected_format,
          "timestamp": error_data.get("timestamp")
      }

  async def handle_turn_notification(self, message: Dict[str, Any]) -> None:
      """Handle turn notification from server.

      Args:
          message: Turn notification message
      """
      turn = message.get("data", {}).get("turn", "unknown")
      logger.info(f"Turn notification: {turn}")

  async def handle_pong(self, message: Dict[str, Any]) -> None:
      """Handle pong response from server.

      Args:
          message: Pong message
      """
      logger.debug("Received pong")

  async def handle_conn_ping(self, message: Dict[str, Any]) -> None:
      """Handle PACKET_CONN_PING from civserver and respond with PACKET_CONN_PONG.

      This is critical for keeping the connection alive. The civserver sends
      periodic ping packets, and if we don't respond with pong, it will
      disconnect the client.

      Args:
          message: Ping message from civserver with structure:
              {
                  "type": "conn_ping",
                  "data": {...}
              }
      """
      logger.debug("Received PACKET_CONN_PING from civserver, responding with PACKET_CONN_PONG")

      # Create pong response
      pong_message = {
          "type": "conn_pong",
          "timestamp": time.time(),
      }

      # Send pong response back to server
      if self.client and self.client.connection_manager:
          try:
              await self.client.connection_manager.send_message(json.dumps(pong_message))
              logger.debug("Successfully sent PACKET_CONN_PONG to civserver")
          except Exception as e:
              logger.error(f"Failed to send PACKET_CONN_PONG: {e}")

  async def handle_game_ready(self, message: Dict[str, Any]) -> None:
      """Handle game_ready message from server.

      This message is sent when the FreeCiv server has fully initialized
      the game with nations assigned, units created, and gameplay ready to begin.

      Args:
          message: game_ready message from server
      """
      # Extract game_id - handle both nested and flat message structures
      game_id = message.get("game_id") or message.get("data", {}).get("game_id", "unknown")

      logger.info(f"🎮 Game ready signal received for game {game_id}")

      # Set flag and event on client to indicate game is ready
      if self.client:
          self.client.game_ready = True
          self.client.game_ready_event.set()  # Wake any coroutines waiting on this event
          logger.info("✅ Game fully initialized - nations assigned, units created, ready to play")
          logger.info(f"   Game ready event set at timestamp {time.time():.1f}")

  async def handle_error(self, message: Dict[str, Any]) -> None:
      """Handle error messages from server.

      Stores error details for debugging and allows retry logic to access them.

      Args:
          message: Error message with structure:
              {
                  "type": "error",
                  "data": {
                      "code": "E122",
                      "message": "Player not assigned yet",
                      "details": {...}
                  }
              }
      """
      error_data = message.get("data", {})
      error_code = error_data.get("code", "UNKNOWN")
      error_message = error_data.get("message", "Unknown error")
      error_details = error_data.get("details", {})

      logger.error(
          f"❌ Server error [{error_code}]: {error_message}\n"
          f"   Details: {error_details}"
      )

      # Store last error in client for get_state() retry logic to access
      if self.client:
          self.client.last_error = {
              "code": error_code,
              "message": error_message,
              "details": error_details,
              "timestamp": time.time()
          }


class ProtocolTranslator:
  """Translates between Game Arena and FreeCiv protocol formats."""

  def to_freeciv_packet(self, action: FreeCivAction) -> Dict[str, Any]:
      """Convert FreeCivAction to proxy's expected action format.

      The proxy expects actions in this format:
      {'action_type': '...', 'actor_id': ..., 'target': {...}, 'parameters': {...}}

      NOT the FreeCiv packet format! The proxy handles the packet conversion internally.

      Args:
          action: FreeCivAction to convert

      Returns:
          Action dictionary compatible with proxy's action validator
      """
      # Build base action structure
      packet = {
          "action_type": action.action_type,
          "actor_id": action.actor_id,
      }

      # Add target if present (including empty dicts)
      if action.target is not None:
          packet["target"] = action.target

      # Add parameters if present (including empty dicts)
      if action.parameters is not None:
          packet["parameters"] = action.parameters

      return packet

  def from_freeciv_packet(self, packet: Dict[str, Any]) -> Dict[str, Any]:
      """Convert FreeCiv packet to Game Arena format.

      Args:
          packet: FreeCiv packet dictionary

      Returns:
          Game Arena format dictionary
      """
      # For now, pass through as-is since FreeCiv3D proxy
      # already sends in a compatible format
      return packet


class MessageQueue:
  """Priority message queue for outgoing messages."""

  def __init__(self):
      """Initialize message queue."""
      self.priority_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
      self.normal_queue: asyncio.Queue = asyncio.Queue()

  async def enqueue(self, message: Dict[str, Any], priority: int = 1) -> None:
      """Add message to queue.

      Args:
          message: Message to queue
          priority: Priority level (0=highest, 1=normal, 2=lowest)
      """
      if priority == 0:
          await self.priority_queue.put((priority, time.time(), message))
      else:
          await self.normal_queue.put(message)

  async def get_next_message(self) -> Optional[Dict[str, Any]]:
      """Get next message from queue.

      Returns:
          Next message or None if empty
      """
      # Check priority queue first
      try:
          _, _, message = self.priority_queue.get_nowait()
          return message
      except asyncio.QueueEmpty:
          pass

      # Then check normal queue
      try:
          message = self.normal_queue.get_nowait()
          return message
      except asyncio.QueueEmpty:
          return None

  async def get_next_message_nowait(self) -> Optional[Dict[str, Any]]:
      """Get next message without waiting.

      Returns:
          Next message or None if empty
      """
      try:
          # Check priority queue first
          if not self.priority_queue.empty():
              _, _, message = self.priority_queue.get_nowait()
              return message

          # Then normal queue
          if not self.normal_queue.empty():
              return self.normal_queue.get_nowait()

      except asyncio.QueueEmpty:
          pass

      return None

  async def process_messages(
      self, processor: Callable, max_messages: int = 10
  ) -> None:
      """Process queued messages.

      Args:
          processor: Async function to process messages
          max_messages: Maximum messages to process per call
      """
      processed = 0
      while processed < max_messages:
          message = await self.get_next_message_nowait()
          if not message:
              break

          try:
              await processor(message)
              processed += 1
          except Exception as e:
              logger.error(f"Error processing message: {e}")
              break
