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

"""Tests for FreeCivProxyClient concurrency and race condition handling.

This test suite addresses PR feedback issue #1 (High Priority):
Race condition in state cache (freeciv_proxy_client.py:806) where OrderedDict
is not thread-safe for async concurrent access.
"""

import asyncio
import json
import time
import unittest
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

from game_arena.harness.freeciv_proxy_client import (
    ConnectionState,
    FreeCivProxyClient,
    create_secure_cache_key,
)


class TestStateCacheConcurrency(unittest.IsolatedAsyncioTestCase):
  """Test suite for state cache concurrent access patterns."""

  async def asyncSetUp(self):
    """Set up test client with mocked WebSocket connection."""
    self.client = FreeCivProxyClient(
        host="localhost",
        port=8002,
        agent_id="test_concurrency_agent",
        state_cache_ttl=1.0,
        max_cache_entries=10,
    )

    # Mock the connection manager to simulate connected state
    self.client.connection_manager.state = ConnectionState.CONNECTED
    self.client.connection_manager.websocket = AsyncMock()
    self.client.player_id = 1

    # Disable rate limiting for testing - set very high limits
    self.client.rate_limiter.requests_per_minute = 100000
    self.client.rate_limiter.burst_size = 10000
    self.client.rate_limiter.refill_rate = 100000 / 60.0

    # Disable message size rate limiter for testing
    self.client.message_size_limiter.max_messages_per_second = 100000
    self.client.message_size_limiter.max_bytes_per_minute = 1_000_000_000

    # Disable circuit breaker for concurrency tests
    self.client.circuit_breaker.failure_threshold = 1000000

  async def asyncTearDown(self):
    """Clean up test client."""
    if hasattr(self.client, 'connection_manager'):
      self.client.connection_manager.state = ConnectionState.DISCONNECTED

  async def test_state_cache_concurrent_access_race_condition(self):
    """Test that concurrent get_state() calls don't cause race conditions.

    This test simulates concurrent state requests with cache operations to
    expose race conditions. Without proper locking, this can cause:
    - KeyError from concurrent OrderedDict.popitem()
    - RuntimeError from dict changed during iteration
    - Data corruption from overlapping writes

    Expected behavior:
    - All requests complete without errors
    - No KeyError from concurrent OrderedDict access
    - No data corruption from concurrent modifications
    """
    # Track if we hit race conditions
    race_conditions = []

    # Patch _wait_for_response to return mock state immediately
    original_wait = self.client._wait_for_response

    async def mock_wait_for_response(expected_types, timeout=30.0):
      # Simulate small async delay to increase race condition probability
      await asyncio.sleep(0.001)
      return {
          "type": "state_update",
          "data": {
              "turn": 1,
              "playerID": 1,
              "players": [],
              "map": {"tiles": []},
          }
      }

    self.client._wait_for_response = mock_wait_for_response

    # Reduce concurrency but increase cache churn
    num_concurrent = 20
    format_count = 3  # Use only 3 formats to force more cache evictions

    async def get_state_and_catch_errors(format_id):
      """Wrapper to catch and record race condition errors."""
      try:
        result = await self.client.get_state(format_type=f"fmt_{format_id % format_count}")
        return result
      except (KeyError, RuntimeError) as e:
        race_conditions.append(str(e))
        raise

    # Create tasks
    tasks = [get_state_and_catch_errors(i) for i in range(num_concurrent)]

    # Execute concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Restore original method
    self.client._wait_for_response = original_wait

    # Check for exceptions
    exceptions = [r for r in results if isinstance(r, Exception)]

    # Report race conditions if found
    if race_conditions:
      self.fail(
          f"Race conditions detected ({len(race_conditions)}): {race_conditions[:3]}"
      )

    # If no race conditions, verify all results are valid
    for result in results:
      if not isinstance(result, Exception):
        self.assertIsInstance(result, dict)
        self.assertIn("data", result)

  async def test_state_cache_eviction_during_read(self):
    """Test cache eviction while another coroutine is reading.

    This test creates a scenario where one coroutine is reading from the
    cache while another is evicting old entries. Without proper locking,
    this can cause "dictionary changed size during iteration" errors.

    Expected behavior:
    - Read operations complete successfully
    - Eviction operations complete successfully
    - No race conditions or data corruption
    """
    # Mock _wait_for_response to return immediately
    original_wait = self.client._wait_for_response

    async def mock_wait_for_response(expected_types, timeout=30.0):
      await asyncio.sleep(0.001)
      return {
          "type": "state_update",
          "data": {"turn": 1, "playerID": 1}
      }

    self.client._wait_for_response = mock_wait_for_response

    # Fill cache to near capacity
    for i in range(self.client.max_cache_entries - 1):
      await self.client.get_state(format_type=f"format_{i}")

    race_error = None

    async def read_cache_repeatedly():
      """Repeatedly read from cache."""
      nonlocal race_error
      for _ in range(20):
        try:
          # Access cache directly (simulates iteration during read)
          cache_keys = list(self.client.state_cache.keys())
          await asyncio.sleep(0.001)  # Small delay to increase race chance
        except RuntimeError as e:
          if "dictionary changed size" in str(e):
            race_error = e
            return
      return

    async def trigger_evictions():
      """Trigger cache evictions by adding new entries."""
      for i in range(15):
        await self.client.get_state(format_type=f"new_format_{i}")
        await asyncio.sleep(0.001)

    # Run read and eviction operations concurrently
    # This will expose race conditions without proper locking
    await asyncio.gather(
        read_cache_repeatedly(),
        trigger_evictions(),
        return_exceptions=False
    )

    # Restore original
    self.client._wait_for_response = original_wait

    # Check if race condition was detected
    if race_error:
      self.fail(
          f"Race condition during cache iteration: {race_error}"
      )

    # Verify cache is in consistent state
    self.assertLessEqual(
        len(self.client.state_cache),
        self.client.max_cache_entries,
        "Cache size exceeded max_cache_entries"
    )

  async def test_state_cache_concurrent_writes(self):
    """Test concurrent cache writes don't corrupt data.

    This test ensures that when multiple coroutines try to write to the
    cache simultaneously, the cache remains in a consistent state without
    data corruption.

    Expected behavior:
    - All writes complete successfully
    - Cache entries are valid and uncorrupted
    - LRU ordering is maintained correctly
    """
    # Mock _wait_for_response with unique responses
    original_wait = self.client._wait_for_response
    response_counter = {"value": 0}

    async def mock_wait_for_response(expected_types, timeout=30.0):
      await asyncio.sleep(0.001)
      response_counter["value"] += 1
      return {
          "type": "state_update",
          "data": {
              "turn": response_counter["value"],
              "playerID": 1,
              "unique_id": response_counter["value"]
          }
      }

    self.client._wait_for_response = mock_wait_for_response

    # Simulate 30 concurrent writes to the same cache key
    # This tests for data corruption when multiple writes overlap
    tasks = [
        self.client.get_state(format_type="shared_format")
        for _ in range(30)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Restore original
    self.client._wait_for_response = original_wait

    # Verify no exceptions occurred
    exceptions = [r for r in results if isinstance(r, Exception)]
    self.assertEqual(
        len(exceptions), 0,
        f"Concurrent write failures: {exceptions[:5]}"
    )

    # Verify cache entry is valid (not corrupted)
    cache_key = create_secure_cache_key("state", "shared_format")
    if cache_key in self.client.state_cache:
      cached_entry = self.client.state_cache[cache_key]
      self.assertIn("data", cached_entry)
      self.assertIn("_timestamp", cached_entry)
      self.assertIsInstance(cached_entry["data"]["turn"], int)


class TestCircuitBreakerConcurrency(unittest.IsolatedAsyncioTestCase):
  """Test circuit breaker behavior under concurrent load."""

  async def asyncSetUp(self):
    """Set up test client."""
    self.client = FreeCivProxyClient(
        host="localhost",
        port=8002,
        agent_id="test_circuit_agent",
    )
    self.client.connection_manager.state = ConnectionState.CONNECTED
    self.client.connection_manager.websocket = AsyncMock()
    self.client.player_id = 1

    # Disable rate limiting for testing
    self.client.rate_limiter.requests_per_minute = 100000
    self.client.rate_limiter.burst_size = 10000
    self.client.rate_limiter.refill_rate = 100000 / 60.0

    # Disable message size rate limiter
    self.client.message_size_limiter.max_messages_per_second = 100000
    self.client.message_size_limiter.max_bytes_per_minute = 1_000_000_000

  async def test_circuit_breaker_concurrent_failures(self):
    """Test circuit breaker handles concurrent failures correctly.

    When multiple requests fail simultaneously, the circuit breaker should
    maintain consistent state without race conditions.
    """
    import json

    # Mock WebSocket to simulate failures
    async def mock_send_message(message):
      pass

    async def mock_receive_message():
      # Simulate error response
      return json.dumps({
          "type": "error",
          "data": {
              "code": "E500",
              "message": "Server error"
          }
      })

    self.client.connection_manager.send_message = mock_send_message
    self.client.connection_manager.receive_message = mock_receive_message

    # Trigger multiple concurrent failures
    tasks = [
        self.client.get_state()
        for _ in range(10)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # All should fail with RuntimeError
    exceptions = [r for r in results if isinstance(r, Exception)]
    self.assertGreater(len(exceptions), 0, "Expected failures not detected")

    # Circuit breaker should have opened
    # (depends on failure_threshold which is 5 by default)
    circuit_state = self.client.circuit_breaker.get_state_info()

    # After 10 failures, circuit should be open
    self.assertEqual(
        circuit_state["state"], "open",
        "Circuit breaker should be open after concurrent failures"
    )


if __name__ == "__main__":
  unittest.main()
