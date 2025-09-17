# Copyright 2025 The game_arena Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Thread-safe LRU cache implementation for FreeCiv parsers."""

import threading
import time
from collections import OrderedDict
from typing import Any, Generic, Optional, TypeVar

K = TypeVar('K')  # Key type
V = TypeVar('V')  # Value type


class LRUCache(Generic[K, V]):
  """Thread-safe LRU (Least Recently Used) cache implementation.

  This cache maintains items in order of access, automatically evicting
  the least recently used items when the cache exceeds its maximum size.

  Features:
    - Thread-safe operations using RLock
    - LRU eviction policy
    - Cache hit/miss statistics
    - Optional TTL (time-to-live) support
    - Clear and invalidation methods

  Examples:
    >>> cache = LRUCache[str, int](max_size=100)
    >>> cache.set("key1", 42)
    >>> value = cache.get("key1")  # Returns 42
    >>> cache.get("missing")       # Returns None
    >>> cache.get("missing", -1)   # Returns -1 (default)
  """

  def __init__(self, max_size: int = 1000, ttl_seconds: Optional[float] = None):
    """Initialize LRU cache.

    Args:
      max_size: Maximum number of items to store
      ttl_seconds: Optional time-to-live in seconds for cache entries

    Raises:
      ValueError: If max_size is not positive
    """
    if max_size <= 0:
      raise ValueError("max_size must be positive")

    self._max_size = max_size
    self._ttl_seconds = ttl_seconds
    self._cache: OrderedDict[K, tuple[V, float]] = OrderedDict()
    self._lock = threading.RLock()

    # Statistics
    self._hits = 0
    self._misses = 0
    self._evictions = 0

  def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
    """Get value from cache, updating access order.

    Args:
      key: Cache key to look up
      default: Default value if key not found

    Returns:
      Cached value if found and not expired, default otherwise
    """
    with self._lock:
      if key not in self._cache:
        self._misses += 1
        return default

      value, timestamp = self._cache[key]

      # Check TTL expiration
      if self._ttl_seconds is not None:
        age = time.time() - timestamp
        if age > self._ttl_seconds:
          del self._cache[key]
          self._misses += 1
          return default

      # Move to end (most recently used)
      self._cache.move_to_end(key)
      self._hits += 1
      return value

  def set(self, key: K, value: V) -> None:
    """Set value in cache, evicting oldest items if necessary.

    Args:
      key: Cache key
      value: Value to store
    """
    with self._lock:
      current_time = time.time()

      if key in self._cache:
        # Update existing key
        self._cache[key] = (value, current_time)
        self._cache.move_to_end(key)
      else:
        # Add new key, evict if necessary
        if len(self._cache) >= self._max_size:
          self._cache.popitem(last=False)  # Remove oldest
          self._evictions += 1

        self._cache[key] = (value, current_time)

  def clear(self) -> None:
    """Clear all cache entries."""
    with self._lock:
      self._cache.clear()

  def invalidate(self, key: K) -> bool:
    """Remove specific key from cache.

    Args:
      key: Key to remove

    Returns:
      True if key was found and removed, False otherwise
    """
    with self._lock:
      if key in self._cache:
        del self._cache[key]
        return True
      return False

  def invalidate_prefix(self, prefix: str) -> int:
    """Remove all keys starting with given prefix.

    Args:
      prefix: Prefix to match for removal

    Returns:
      Number of keys removed
    """
    with self._lock:
      keys_to_remove = [
          key for key in self._cache
          if str(key).startswith(prefix)
      ]
      for key in keys_to_remove:
        del self._cache[key]
      return len(keys_to_remove)

  def cleanup_expired(self) -> int:
    """Remove expired entries based on TTL.

    Returns:
      Number of expired entries removed
    """
    if self._ttl_seconds is None:
      return 0

    with self._lock:
      current_time = time.time()
      expired_keys = []

      for key, (value, timestamp) in self._cache.items():
        age = current_time - timestamp
        if age > self._ttl_seconds:
          expired_keys.append(key)

      for key in expired_keys:
        del self._cache[key]

      return len(expired_keys)

  def __len__(self) -> int:
    """Return current cache size."""
    with self._lock:
      return len(self._cache)

  def __contains__(self, key: K) -> bool:
    """Check if key exists in cache (without updating access order)."""
    with self._lock:
      if key not in self._cache:
        return False

      # Check TTL without updating access order
      if self._ttl_seconds is not None:
        _, timestamp = self._cache[key]
        age = time.time() - timestamp
        if age > self._ttl_seconds:
          del self._cache[key]
          return False

      return True

  @property
  def max_size(self) -> int:
    """Maximum cache size."""
    return self._max_size

  @property
  def hit_rate(self) -> float:
    """Cache hit rate as a percentage (0.0-100.0)."""
    with self._lock:
      total_requests = self._hits + self._misses
      if total_requests == 0:
        return 0.0
      return (self._hits / total_requests) * 100.0

  @property
  def statistics(self) -> dict[str, Any]:
    """Cache statistics for monitoring and debugging."""
    with self._lock:
      total_requests = self._hits + self._misses
      return {
          'size': len(self._cache),
          'max_size': self._max_size,
          'hits': self._hits,
          'misses': self._misses,
          'evictions': self._evictions,
          'total_requests': total_requests,
          'hit_rate_percent': self.hit_rate,
          'ttl_seconds': self._ttl_seconds
      }

  def reset_statistics(self) -> None:
    """Reset cache statistics counters."""
    with self._lock:
      self._hits = 0
      self._misses = 0
      self._evictions = 0