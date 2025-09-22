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

"""Comprehensive error recovery tests for FreeCiv LLM Agent."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from game_arena.harness.freeciv_llm_agent import FreeCivLLMAgent
from game_arena.harness.freeciv_proxy_client import FreeCivProxyClient
from game_arena.harness.freeciv_state import FreeCivAction, FreeCivState
from game_arena.harness.tests.test_helpers import (
    MockModelWithPredictableResponses,
    FreeCivTestData
)


class TestFreeCivErrorRecovery(unittest.IsolatedAsyncioTestCase):
  """Test error recovery mechanisms in FreeCiv LLM Agent."""

  def setUp(self):
    """Set up test fixtures."""
    self.mock_model = MockModelWithPredictableResponses()
    self.valid_observation = FreeCivTestData.create_sample_observation(player_id=1)
    self.valid_state = FreeCivTestData.create_sample_freeciv_state(player_id=1)

  def test_model_api_failure_recovery(self):
    """Test recovery from model API failures."""
    # Create model that fails initially then succeeds
    failing_model = Mock()
    failing_model.model_name = "test-model"
    call_count = 0

    def side_effect(*args, **kwargs):
      nonlocal call_count
      call_count += 1
      if call_count <= 2:  # Fail first 2 calls
        raise Exception("API rate limit exceeded")
      return "unit_move_warrior(101)_to(11,14)"

    failing_model.return_value = side_effect
    failing_model.side_effect = side_effect

    agent = FreeCivLLMAgent(
        model=failing_model,
        fallback_to_random=True,
        max_retries=3
    )

    # Should recover after retries
    result = agent(self.valid_observation, {})
    self.assertIsInstance(result, dict)
    self.assertIn("submission", result)

  def test_invalid_observation_handling(self):
    """Test handling of invalid observation data."""
    agent = FreeCivLLMAgent(
        model=self.mock_model,
        fallback_to_random=True
    )

    test_cases = [
      {},  # Empty observation
      {"invalid": "data"},  # Missing required fields
      {"turn": "not_a_number"},  # Invalid data types
      {"playerID": -1, "turn": 1},  # Invalid player ID
      None,  # Null observation
    ]

    for invalid_obs in test_cases:
      with self.subTest(observation=invalid_obs):
        try:
          result = agent(invalid_obs, {})
          # If it succeeds, should return valid format
          self.assertIsInstance(result, dict)
          self.assertIn("submission", result)
        except (ValueError, KeyError, TypeError, AttributeError):
          # Acceptable controlled failure
          pass

  def test_empty_legal_actions_recovery(self):
    """Test handling of empty legal actions."""
    agent = FreeCivLLMAgent(
        model=self.mock_model,
        fallback_to_random=True
    )

    # Observation with no legal actions
    empty_obs = self.valid_observation.copy()
    empty_obs["legalActions"] = []

    try:
      result = agent(empty_obs, {})
      # Should handle gracefully or fail with controlled exception
      if result:
        self.assertIsInstance(result, dict)
        self.assertIn("submission", result)
    except ValueError:
      # Acceptable to fail with empty legal actions
      pass

  def test_malformed_legal_actions_recovery(self):
    """Test handling of malformed legal actions."""
    agent = FreeCivLLMAgent(
        model=self.mock_model,
        fallback_to_random=True
    )

    test_cases = [
      {"legalActions": None},  # Null legal actions
      {"legalActions": "not_a_list"},  # Wrong type
      {"legalActions": [None, "invalid"]},  # Mixed invalid data
      {"legalActions": [-1, -2, -3]},  # Negative action IDs
    ]

    for malformed_obs in test_cases:
      obs = self.valid_observation.copy()
      obs.update(malformed_obs)

      with self.subTest(observation=obs):
        try:
          result = agent(obs, {})
          if result:
            self.assertIsInstance(result, dict)
            self.assertIn("submission", result)
        except (ValueError, TypeError, AttributeError):
          # Acceptable controlled failure
          pass

  def test_model_timeout_recovery(self):
    """Test recovery from model timeouts."""
    # Mock model that takes too long
    slow_model = Mock()
    slow_model.model_name = "slow-model"

    def slow_response(*args, **kwargs):
      import time
      time.sleep(2)  # Simulate slow response
      return "unit_move_warrior(101)_to(11,14)"

    slow_model.return_value = slow_response
    slow_model.side_effect = slow_response

    agent = FreeCivLLMAgent(
        model=slow_model,
        fallback_to_random=True,
        model_timeout=1.0  # 1 second timeout
    )

    # Should timeout and fallback
    result = agent(self.valid_observation, {})
    self.assertIsInstance(result, dict)
    self.assertIn("submission", result)

  def test_action_parser_failure_recovery(self):
    """Test recovery from action parser failures."""
    # Model that returns unparseable responses
    unparseable_model = Mock()
    unparseable_model.model_name = "unparseable-model"
    unparseable_model.return_value = "completely invalid response format!!!"
    unparseable_model.side_effect = None

    agent = FreeCivLLMAgent(
        model=unparseable_model,
        fallback_to_random=True,
        use_rethinking=True,
        max_rethinks=2
    )

    # Should fallback to random action
    result = agent(self.valid_observation, {})
    self.assertIsInstance(result, dict)
    self.assertIn("submission", result)
    self.assertIn(result["submission"], self.valid_observation["legalActions"])

  def test_memory_corruption_recovery(self):
    """Test recovery from memory corruption."""
    agent = FreeCivLLMAgent(
        model=self.mock_model,
        memory_size=3
    )

    # Corrupt memory with invalid data
    agent.memory.history = [
      {"invalid": "data"},
      None,
      {"action": "corrupted"},
    ]

    # Should still function despite corrupted memory
    result = agent(self.valid_observation, {})
    self.assertIsInstance(result, dict)
    self.assertIn("submission", result)

  def test_proxy_client_connection_failure(self):
    """Test handling of proxy client connection failures."""
    # Mock proxy client that fails to connect
    failing_client = AsyncMock()
    failing_client.is_connected.return_value = False
    failing_client.connect.side_effect = ConnectionError("Cannot connect to server")
    failing_client.get_game_state.side_effect = ConnectionError("Not connected")

    agent = FreeCivLLMAgent(
        model=self.mock_model,
        proxy_client=failing_client,
        fallback_to_random=True
    )

    # Should handle connection failures gracefully
    result = agent(self.valid_observation, {})
    self.assertIsInstance(result, dict)
    self.assertIn("submission", result)

  async def test_websocket_disconnection_recovery(self):
    """Test recovery from WebSocket disconnections."""
    # Mock client that disconnects unexpectedly
    disconnecting_client = AsyncMock()
    disconnecting_client.is_connected.side_effect = [True, False, True]  # Disconnect then reconnect

    async def get_state_with_disconnection():
      if disconnecting_client.is_connected():
        return self.valid_observation
      else:
        raise ConnectionError("WebSocket disconnected")

    disconnecting_client.get_game_state.side_effect = get_state_with_disconnection

    agent = FreeCivLLMAgent(
        model=self.mock_model,
        proxy_client=disconnecting_client
    )

    # Should attempt recovery
    try:
      result = agent(self.valid_observation, {})
      self.assertIsInstance(result, dict)
    except ConnectionError:
      # Acceptable if recovery fails
      pass

  def test_state_validation_failure_recovery(self):
    """Test recovery from state validation failures."""
    agent = FreeCivLLMAgent(
        model=self.mock_model,
        fallback_to_random=True
    )

    # Create observation with invalid state structure
    invalid_state_obs = {
      "turn": 1,
      "playerID": 1,
      "legalActions": [0, 1, 2],
      "players": [{"id": 1}],  # Missing required player fields
      "map": {"width": -1},  # Invalid map data
    }

    # Should handle invalid state gracefully
    result = agent(invalid_state_obs, {})
    self.assertIsInstance(result, dict)
    self.assertIn("submission", result)

  def test_concurrent_access_safety(self):
    """Test thread safety under concurrent access."""
    agent = FreeCivLLMAgent(
        model=self.mock_model,
        memory_size=5
    )

    import threading
    import queue

    results = queue.Queue()
    errors = queue.Queue()

    def worker():
      try:
        result = agent(self.valid_observation, {})
        results.put(result)
      except Exception as e:
        errors.put(e)

    # Create multiple threads accessing agent concurrently
    threads = []
    for _ in range(5):
      thread = threading.Thread(target=worker)
      threads.append(thread)

    # Start all threads
    for thread in threads:
      thread.start()

    # Wait for completion
    for thread in threads:
      thread.join(timeout=10)

    # Check results
    result_count = results.qsize()
    error_count = errors.qsize()

    # At least some should succeed
    self.assertGreater(result_count, 0)

    # Check that successful results are valid
    while not results.empty():
      result = results.get()
      self.assertIsInstance(result, dict)
      self.assertIn("submission", result)

  def test_resource_exhaustion_recovery(self):
    """Test recovery from resource exhaustion scenarios."""
    agent = FreeCivLLMAgent(
        model=self.mock_model,
        memory_size=1000  # Large memory to test limits
    )

    # Fill memory with many entries
    for i in range(500):
      mock_action = FreeCivAction(
          action_type="unit_move",
          actor_id=100 + i,
          target={"x": 10, "y": 10},
          parameters={},
          source="unit"
      )
      mock_result = {"success": True, "score_delta": 1}
      agent.memory.record_action(mock_action, mock_result)

    # Should still function with full memory
    result = agent(self.valid_observation, {})
    self.assertIsInstance(result, dict)
    self.assertIn("submission", result)

    # Memory should have been trimmed
    self.assertLessEqual(len(agent.memory.history), agent.memory.max_size)

  def test_circular_dependency_prevention(self):
    """Test prevention of circular dependencies in rethinking."""
    agent = FreeCivLLMAgent(
        model=self.mock_model,
        use_rethinking=True,
        max_rethinks=10  # High limit
    )

    # Mock action parser that always returns illegal moves
    with patch.object(agent.action_parser, 'parse') as mock_parse:
      mock_parse.return_value = None  # Always invalid

      # Should eventually give up and fallback
      result = agent(self.valid_observation, {})
      self.assertIsInstance(result, dict)
      self.assertIn("submission", result)

      # Should have limited retries, not infinite
      self.assertLessEqual(mock_parse.call_count, agent.max_rethinks + 1)

  def test_strategy_adaptation_failure_recovery(self):
    """Test recovery from strategy adaptation failures."""
    agent = FreeCivLLMAgent(
        model=self.mock_model,
        strategy="balanced"
    )

    # Mock strategy manager that fails
    with patch.object(agent.strategy_manager, 'adapt_strategy') as mock_adapt:
      mock_adapt.side_effect = Exception("Strategy adaptation failed")

      # Should still function with original strategy
      result = agent(self.valid_observation, {})
      self.assertIsInstance(result, dict)
      self.assertIn("submission", result)

      # Strategy should remain unchanged
      self.assertEqual(agent.strategy, "balanced")

  def test_telemetry_failure_isolation(self):
    """Test that telemetry failures don't affect core functionality."""
    agent = FreeCivLLMAgent(
        model=self.mock_model,
        enable_telemetry=True
    )

    # Mock telemetry to fail
    if hasattr(agent, 'telemetry_manager'):
      with patch.object(agent.telemetry_manager, 'track_action') as mock_telemetry:
        mock_telemetry.side_effect = Exception("Telemetry error")

        # Core functionality should still work
        result = agent(self.valid_observation, {})
        self.assertIsInstance(result, dict)
        self.assertIn("submission", result)

  def test_graceful_degradation_modes(self):
    """Test various graceful degradation modes."""
    test_modes = [
      {"use_rethinking": False, "fallback_to_random": True},
      {"use_rethinking": True, "fallback_to_random": False},
      {"memory_size": 0, "enable_telemetry": False},
      {"strategy": None, "use_rethinking": False},
    ]

    for mode in test_modes:
      with self.subTest(mode=mode):
        try:
          agent = FreeCivLLMAgent(model=self.mock_model, **mode)
          result = agent(self.valid_observation, {})
          self.assertIsInstance(result, dict)
          self.assertIn("submission", result)
        except Exception as e:
          # Some degraded modes might not be supported
          self.assertIsInstance(e, (ValueError, TypeError))


if __name__ == '__main__':
  unittest.main()