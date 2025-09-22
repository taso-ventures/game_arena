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

"""Resilience and stress tests for FreeCiv LLM Agent components."""

import asyncio
import random
import threading
import time
import unittest
from unittest.mock import Mock, patch

from game_arena.harness.freeciv_llm_agent import FreeCivLLMAgent
from game_arena.harness.freeciv_memory import GameMemory, TokenManager
from game_arena.harness.freeciv_state import FreeCivAction
from game_arena.harness.tests.test_helpers import (
    MockModelWithPredictableResponses,
    FreeCivTestData
)


class TestFreeCivResilience(unittest.TestCase):
  """Test resilience and stress scenarios for FreeCiv components."""

  def setUp(self):
    """Set up test fixtures."""
    self.mock_model = MockModelWithPredictableResponses()
    self.valid_observation = FreeCivTestData.create_sample_observation(player_id=1)

  def test_memory_stress_load(self):
    """Test memory system under high load."""
    memory = GameMemory(max_size=100)

    # Add many actions rapidly
    start_time = time.time()
    for i in range(1000):
      action = FreeCivAction(
          action_type="unit_move",
          actor_id=100 + i,
          target={"x": random.randint(0, 50), "y": random.randint(0, 50)},
          parameters={},
          source="unit"
      )
      result = {
        "success": random.choice([True, False]),
        "score_delta": random.randint(-10, 10)
      }
      memory.record_action(action, result)

    elapsed_time = time.time() - start_time

    # Should complete quickly
    self.assertLess(elapsed_time, 5.0)

    # Memory should be properly bounded
    self.assertLessEqual(len(memory.history), memory.max_size)

    # Should still function correctly
    context = memory.get_context(max_tokens=500)
    self.assertIsInstance(context, str)

    performance = memory.get_performance_summary()
    self.assertIsInstance(performance, dict)
    self.assertEqual(performance["total_actions"], 1000)

  def test_token_manager_stress(self):
    """Test token manager with various text patterns."""
    token_manager = TokenManager("gpt-4")

    # Test with various text patterns
    test_texts = [
      "",  # Empty string
      "a" * 10000,  # Very long repetitive text
      "🎮" * 1000,  # Unicode characters
      "\n\n\n" * 500,  # Lots of newlines
      "Hello " * 2000,  # Repeated patterns
      json_text := '{"action": "move", "target": {"x": 10, "y": 15}}' * 100,  # JSON
      code_text := 'def function(x, y):\n    return x + y\n' * 200,  # Code
      "FreeCiv unit_move_warrior(101)_to(11,14) " * 300,  # Game actions
    ]

    for i, text in enumerate(test_texts):
      with self.subTest(text_type=f"pattern_{i}"):
        # Should handle all patterns without crashing
        tokens = token_manager.count_tokens(text)
        self.assertIsInstance(tokens, int)
        self.assertGreaterEqual(tokens, 0)

        # Test truncation
        if len(text) > 100:
          truncated = token_manager.truncate_to_limit(text, reserve=10)
          self.assertIsInstance(truncated, str)
          self.assertLessEqual(len(truncated), len(text))

  def test_concurrent_agent_access(self):
    """Test multiple threads accessing agent simultaneously."""
    agent = FreeCivLLMAgent(
        model=self.mock_model,
        memory_size=10
    )

    results = []
    errors = []
    lock = threading.Lock()

    def worker(worker_id):
      try:
        # Simulate varying observation data
        obs = self.valid_observation.copy()
        obs["turn"] = worker_id
        obs["playerID"] = (worker_id % 2) + 1

        result = agent(obs, {})

        with lock:
          results.append(result)

      except Exception as e:
        with lock:
          errors.append((worker_id, e))

    # Create many concurrent workers
    threads = []
    for i in range(20):
      thread = threading.Thread(target=worker, args=(i,))
      threads.append(thread)

    # Start all threads
    start_time = time.time()
    for thread in threads:
      thread.start()

    # Wait for completion
    for thread in threads:
      thread.join(timeout=30)

    elapsed_time = time.time() - start_time

    # Should complete in reasonable time
    self.assertLess(elapsed_time, 30)

    # Most should succeed
    self.assertGreater(len(results), len(errors))

    # All results should be valid
    for result in results:
      self.assertIsInstance(result, dict)
      self.assertIn("submission", result)

  def test_memory_corruption_resilience(self):
    """Test resilience to memory corruption."""
    agent = FreeCivLLMAgent(
        model=self.mock_model,
        memory_size=5
    )

    # Add some valid entries
    for i in range(3):
      action = FreeCivAction(
          action_type="unit_move",
          actor_id=101 + i,
          target={"x": 10, "y": 14},
          parameters={},
          source="unit"
      )
      agent.memory.record_action(action, {"success": True})

    # Corrupt memory with various invalid data
    agent.memory.history.extend([
      None,  # Null entry
      {"invalid": "structure"},  # Wrong structure
      {"action": "corrupted", "result": None},  # Partially corrupted
      [],  # Wrong type
      {"action": FreeCivAction("invalid_type", 999, {}, {}, "invalid"), "result": {}},
    ])

    # Agent should still function
    result = agent(self.valid_observation, {})
    self.assertIsInstance(result, dict)
    self.assertIn("submission", result)

    # Should be able to get context despite corruption
    context = agent.memory.get_context(max_tokens=200)
    self.assertIsInstance(context, str)

  def test_model_response_edge_cases(self):
    """Test handling of various model response edge cases."""
    edge_case_responses = [
      "",  # Empty response
      " \n\t ",  # Whitespace only
      "unit_move_warrior(999999)_to(999,999)",  # Out of bounds action
      "unit_move_warrior(101)_to(11,14)\nunit_move_warrior(102)_to(12,15)",  # Multiple actions
      "I think the best move is to... unit_move_warrior(101)_to(11,14)",  # Explanation + action
      "ERROR: Cannot process request",  # Error message
      "unit_move_warrior(invalid)_to(invalid,invalid)",  # Invalid parameters
      "🎮🚀⚔️",  # Only emojis
      "a" * 10000,  # Very long response
      json_response := '{"action": "unit_move", "details": "complex"}',  # JSON response
    ]

    for i, response in enumerate(edge_case_responses):
      with self.subTest(response_type=f"edge_case_{i}"):
        # Mock model to return specific response
        edge_model = Mock()
        edge_model.model_name = "edge-test-model"
        edge_model.return_value = response
        edge_model.side_effect = None

        agent = FreeCivLLMAgent(
            model=edge_model,
            fallback_to_random=True
        )

        # Should handle gracefully
        result = agent(self.valid_observation, {})
        self.assertIsInstance(result, dict)
        self.assertIn("submission", result)

  def test_rapid_sequential_calls(self):
    """Test rapid sequential calls to agent."""
    agent = FreeCivLLMAgent(
        model=self.mock_model,
        memory_size=5
    )

    results = []
    start_time = time.time()

    # Make rapid sequential calls
    for i in range(50):
      obs = self.valid_observation.copy()
      obs["turn"] = i + 1

      result = agent(obs, {})
      results.append(result)

      # Short delay to simulate rapid gameplay
      time.sleep(0.01)

    elapsed_time = time.time() - start_time

    # Should complete quickly
    self.assertLess(elapsed_time, 10)

    # All should succeed
    self.assertEqual(len(results), 50)

    for result in results:
      self.assertIsInstance(result, dict)
      self.assertIn("submission", result)

    # Memory should show recent actions
    self.assertGreater(len(agent.memory.history), 0)
    self.assertLessEqual(len(agent.memory.history), agent.memory.max_size)

  def test_resource_cleanup(self):
    """Test proper resource cleanup."""
    agents = []

    # Create many agents
    for i in range(10):
      agent = FreeCivLLMAgent(
          model=self.mock_model,
          memory_size=10
      )
      agents.append(agent)

      # Use each agent
      result = agent(self.valid_observation, {})
      self.assertIsInstance(result, dict)

    # Explicitly delete agents
    for agent in agents:
      del agent

    # Should not have memory leaks (hard to test directly)
    # At minimum, this should not crash

  def test_extreme_observation_sizes(self):
    """Test handling of extremely large or small observations."""
    agent = FreeCivLLMAgent(
        model=self.mock_model,
        fallback_to_random=True
    )

    # Minimal observation
    minimal_obs = {
      "turn": 1,
      "playerID": 1,
      "legalActions": [0]
    }

    result = agent(minimal_obs, {})
    self.assertIsInstance(result, dict)
    self.assertIn("submission", result)

    # Large observation with many entities
    large_obs = self.valid_observation.copy()
    large_obs["players"] = []

    # Add many players
    for i in range(100):
      player = {
        "id": i + 1,
        "name": f"Player{i}",
        "nation": f"Nation{i}",
        "score": random.randint(0, 1000),
        "cities": [
          {
            "id": j,
            "name": f"City{j}",
            "x": random.randint(0, 50),
            "y": random.randint(0, 50),
            "population": random.randint(1, 20)
          } for j in range(random.randint(1, 5))
        ],
        "units": [
          {
            "id": k,
            "type": random.choice(["warrior", "settler", "archer"]),
            "x": random.randint(0, 50),
            "y": random.randint(0, 50),
            "health": random.randint(1, 100)
          } for k in range(random.randint(1, 10))
        ]
      }
      large_obs["players"].append(player)

    # Add many legal actions
    large_obs["legalActions"] = list(range(1000))

    # Should handle large observation
    result = agent(large_obs, {})
    self.assertIsInstance(result, dict)
    self.assertIn("submission", result)

  def test_strategy_switching_stress(self):
    """Test rapid strategy switching."""
    agent = FreeCivLLMAgent(
        model=self.mock_model,
        strategy="balanced"
    )

    strategies = ["aggressive_expansion", "economic_focus", "defensive_turtle", "science_victory"]

    # Rapidly switch strategies
    for i in range(20):
      new_strategy = random.choice(strategies)
      agent.strategy = new_strategy

      # Test with each strategy
      result = agent(self.valid_observation, {})
      self.assertIsInstance(result, dict)
      self.assertIn("submission", result)

      # Verify strategy was applied
      self.assertEqual(agent.strategy, new_strategy)

  def test_telemetry_stress_load(self):
    """Test telemetry system under stress."""
    agent = FreeCivLLMAgent(
        model=self.mock_model,
        enable_telemetry=True
    )

    # Generate many actions to stress telemetry
    for i in range(100):
      obs = self.valid_observation.copy()
      obs["turn"] = i + 1

      result = agent(obs, {})
      self.assertIsInstance(result, dict)

    # Telemetry should still be functional
    if hasattr(agent, 'telemetry_manager'):
      metrics = agent.telemetry_manager.get_current_metrics()
      self.assertIsInstance(metrics, dict)

  def test_action_converter_stress(self):
    """Test action converter with many conversions."""
    agent = FreeCivLLMAgent(model=self.mock_model)
    state = FreeCivTestData.create_sample_freeciv_state(player_id=1)

    # Test many action conversions
    for i in range(100):
      action = FreeCivAction(
          action_type=random.choice(["unit_move", "city_production", "player_action"]),
          actor_id=100 + i,
          target={"x": random.randint(0, 50), "y": random.randint(0, 50)},
          parameters={},
          source=random.choice(["unit", "city", "player"])
      )

      # Convert to int and back
      try:
        action_int = agent.action_converter.action_to_int(action, state, player_id=1)
        self.assertIsInstance(action_int, int)

        converted_back = agent.action_converter.int_to_action(action_int, state, player_id=1)
        self.assertIsInstance(converted_back, FreeCivAction)

      except (ValueError, KeyError):
        # Some random actions may not be convertible
        pass

  def test_prompt_builder_stress(self):
    """Test prompt builder with various scenarios."""
    agent = FreeCivLLMAgent(model=self.mock_model)

    # Test with different observation types
    test_scenarios = [
      (self.valid_observation, [0, 1, 2]),  # Normal case
      (self.valid_observation, list(range(100))),  # Many legal actions
      (self.valid_observation, []),  # No legal actions
      ({**self.valid_observation, "turn": 1000}, [0, 1]),  # Late game
      ({**self.valid_observation, "players": []}, [0]),  # No players
    ]

    for obs, legal_actions in test_scenarios:
      with self.subTest(scenario=obs.get("turn", "unknown")):
        try:
          prompt = agent.prompt_builder.build_enhanced_prompt(
              observation=obs,
              legal_actions=legal_actions,
              model_name=self.mock_model.model_name
          )
          self.assertIsInstance(prompt, str)
          self.assertGreater(len(prompt), 0)

        except (ValueError, KeyError):
          # Some invalid scenarios may fail
          pass


if __name__ == '__main__':
  unittest.main()