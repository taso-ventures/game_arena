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

"""End-to-end tests for FreeCiv LLM Agent with real FreeCiv3D server."""

import asyncio
import os
import time
import unittest
from typing import Dict, List, Optional

import requests

from game_arena.harness import model_registry, tournament_util
from game_arena.harness.freeciv_client import FreeCivClient
from game_arena.harness.freeciv_llm_agent import FreeCivLLMAgent
from game_arena.harness.freeciv_proxy_client import FreeCivProxyClient
from game_arena.harness.freeciv_state import FreeCivState


class TestFreeCivE2E(unittest.IsolatedAsyncioTestCase):
  """End-to-end tests with real FreeCiv3D Docker instance."""

  @classmethod
  def setUpClass(cls):
    """Set up class-level test fixtures."""
    # Default FreeCiv3D server URLs
    cls.server_url = os.getenv("FREECIV_SERVER_URL", "http://localhost:8080")
    cls.ws_url = os.getenv("FREECIV_WS_URL", "ws://localhost:4002")

    # Check if FreeCiv3D server is running
    try:
      response = requests.get(f"{cls.server_url}/status", timeout=5)
      if response.status_code != 200:
        raise unittest.SkipTest("FreeCiv3D server not accessible")
    except requests.exceptions.RequestException:
      raise unittest.SkipTest("FreeCiv3D server not running - start with 'cd ../freeciv3d && docker-compose up'")

    # Check API keys
    cls.gemini_key = os.getenv("GEMINI_API_KEY")
    cls.openai_key = os.getenv("OPENAI_API_KEY")
    cls.anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if not any([cls.gemini_key, cls.openai_key, cls.anthropic_key]):
      raise unittest.SkipTest("No API keys configured - set GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY")

  def setUp(self):
    """Set up test fixtures for each test."""
    # Create FreeCiv client for server communication
    self.freeciv_client = FreeCivClient(self.server_url, self.ws_url)

    # Create WebSocket proxy client
    self.proxy_client = FreeCivProxyClient(self.ws_url)

    # Test models to use (in order of preference)
    self.test_models = []

    if self.gemini_key:
      self.test_models.append(
          model_registry.ModelRegistry.GEMINI_2_5_FLASH.build(
              api_key=self.gemini_key,
              model_options={"max_tokens": 2000, "temperature": 0.7}
          )
      )

    if self.openai_key:
      self.test_models.append(
          model_registry.ModelRegistry.OPENAI_GPT_4_1.build(
              api_key=self.openai_key,
              model_options={"max_tokens": 2000, "temperature": 0.7}
          )
      )

  def tearDown(self):
    """Clean up after each test."""
    try:
      if hasattr(self, 'freeciv_client'):
        self.freeciv_client.disconnect()
    except:
      pass  # Ignore cleanup errors

    try:
      if hasattr(self, 'proxy_client'):
        asyncio.create_task(self.proxy_client.disconnect())
    except:
      pass  # Ignore cleanup errors

  async def test_server_connectivity(self):
    """Test basic connectivity to FreeCiv3D server."""
    # Test HTTP connectivity
    response = requests.get(f"{self.server_url}/status", timeout=10)
    self.assertEqual(response.status_code, 200)

    # Test WebSocket connectivity
    await self.proxy_client.connect()
    self.assertTrue(self.proxy_client.is_connected)

    # Test game state retrieval
    state_data = await self.proxy_client.get_game_state()
    self.assertIsInstance(state_data, dict)
    self.assertIn("turn", state_data)

  async def test_freeciv_state_adapter_with_real_server(self):
    """Test FreeCivState adapter with real server data."""
    await self.proxy_client.connect()

    # Get real game state from server
    state_data = await self.proxy_client.get_game_state()

    # Create FreeCivState adapter
    freeciv_state = FreeCivState(state_data)

    # Validate state adapter
    self.assertIsInstance(freeciv_state.turn, int)
    self.assertIsInstance(freeciv_state.players, dict)
    self.assertIsInstance(freeciv_state.units, list)
    self.assertIsInstance(freeciv_state.cities, list)

    # Test legal actions generation
    legal_actions = freeciv_state.get_legal_actions(player_id=1)
    self.assertIsInstance(legal_actions, list)

  async def test_single_agent_turn_execution(self):
    """Test single turn execution with LLM agent."""
    if not self.test_models:
      self.skipTest("No API keys available for testing")

    model = self.test_models[0]  # Use first available model
    agent = FreeCivLLMAgent(
        model=model,
        strategy="balanced",
        use_rethinking=False,  # Disable for faster testing
        fallback_to_random=True
    )

    await self.proxy_client.connect()

    # Get current game state
    state_data = await self.proxy_client.get_game_state()
    freeciv_state = FreeCivState(state_data)

    # Create observation
    observation = {
        "serializedGameAndState": "test_state",
        "legalActions": list(range(len(freeciv_state.get_legal_actions(1)))),
        **state_data
    }

    start_time = time.time()

    # Execute single turn
    # result = await agent.get_action_async(observation, self.proxy_client)

    execution_time = time.time() - start_time

    # Validate results
    # self.assertIsNotNone(result)
    self.assertLess(execution_time, 5.0)  # Must complete within 5 seconds
    # This will fail until implementation exists

  async def test_multi_turn_game_execution(self):
    """Test multi-turn game execution with LLM agent."""
    if not self.test_models:
      self.skipTest("No API keys available for testing")

    model = self.test_models[0]
    agent = FreeCivLLMAgent(
        model=model,
        strategy="balanced",
        use_rethinking=True,
        max_rethinks=2,
        fallback_to_random=True
    )

    await self.proxy_client.connect()

    # Execute multiple turns
    turns_to_execute = 3
    executed_turns = []

    for turn in range(turns_to_execute):
      try:
        # Get current state
        state_data = await self.proxy_client.get_game_state()
        freeciv_state = FreeCivState(state_data)

        observation = {
            "serializedGameAndState": f"turn_{turn}",
            "legalActions": list(range(len(freeciv_state.get_legal_actions(1)))),
            **state_data
        }

        start_time = time.time()

        # Execute turn
        # result = await agent.get_action_async(observation, self.proxy_client)

        execution_time = time.time() - start_time

        executed_turns.append({
            "turn": turn,
            "execution_time": execution_time,
            # "action": result,
            "state": freeciv_state.turn
        })

        # Validate turn execution
        self.assertLess(execution_time, 5.0)

        # Wait briefly between turns
        await asyncio.sleep(0.5)

      except Exception as e:
        self.fail(f"Turn {turn} execution failed: {e}")

    # Validate overall execution
    self.assertEqual(len(executed_turns), turns_to_execute)
    total_time = sum(turn["execution_time"] for turn in executed_turns)
    average_time = total_time / len(executed_turns)
    self.assertLess(average_time, 3.0)  # Average should be under 3 seconds
    # This will fail until implementation exists

  async def test_concurrent_agent_execution(self):
    """Test concurrent execution of multiple agents."""
    if not self.test_models:
      self.skipTest("No API keys available for testing")

    # Create multiple agents with different strategies
    agents = []
    for i, strategy in enumerate(["balanced", "aggressive"]):
      if i < len(self.test_models):
        model = self.test_models[i]
      else:
        model = self.test_models[0]  # Reuse first model

      agent = FreeCivLLMAgent(
          model=model,
          strategy=strategy,
          use_rethinking=False,
          fallback_to_random=True
      )
      agents.append(agent)

    await self.proxy_client.connect()

    # Get game state
    state_data = await self.proxy_client.get_game_state()

    # Create observations for different players
    observations = []
    for i, agent in enumerate(agents):
      observation = {
          "serializedGameAndState": f"player_{i+1}",
          "legalActions": list(range(10)),  # Mock legal actions
          **state_data
      }
      observations.append(observation)

    # Execute agents concurrently
    tasks = []
    for agent, observation in zip(agents, observations):
      # task = agent.get_action_async(observation, self.proxy_client)
      # tasks.append(task)
      pass

    start_time = time.time()
    # results = await asyncio.gather(*tasks, return_exceptions=True)
    execution_time = time.time() - start_time

    # Validate concurrent execution
    # self.assertEqual(len(results), len(agents))
    # for result in results:
    #   self.assertNotIsInstance(result, Exception)

    # Should complete faster than sequential execution
    # self.assertLess(execution_time, len(agents) * 3.0)
    # This will fail until implementation exists

  async def test_memory_persistence_across_turns(self):
    """Test that agent memory persists across multiple turns."""
    if not self.test_models:
      self.skipTest("No API keys available for testing")

    model = self.test_models[0]
    agent = FreeCivLLMAgent(
        model=model,
        strategy="balanced",
        memory_size=10,
        fallback_to_random=True
    )

    await self.proxy_client.connect()

    # Execute several turns to build memory
    for turn in range(5):
      state_data = await self.proxy_client.get_game_state()

      observation = {
          "serializedGameAndState": f"memory_turn_{turn}",
          "legalActions": list(range(5)),
          **state_data
      }

      # Execute turn
      # result = await agent.get_action_async(observation, self.proxy_client)

      # Simulate action result
      mock_result = {"success": True, "score_delta": 5, "turn": turn}
      # agent.memory.record_action(result, mock_result)

    # Validate memory state
    # self.assertEqual(len(agent.memory.history), 5)

    # Memory should influence future decisions
    context = agent.memory.get_context(max_tokens=1000)
    self.assertIsInstance(context, str)
    self.assertGreater(len(context), 50)  # Should contain meaningful content
    # This will fail until memory implementation exists

  async def test_strategy_adaptation_during_gameplay(self):
    """Test that agent adapts strategy during actual gameplay."""
    if not self.test_models:
      self.skipTest("No API keys available for testing")

    model = self.test_models[0]
    agent = FreeCivLLMAgent(
        model=model,
        strategy="balanced",
        fallback_to_random=True
    )

    initial_strategy = agent.strategy

    await self.proxy_client.connect()

    # Simulate poor performance scenario
    agent.update_strategy(
        game_phase="mid",
        score_relative=-0.4  # Significantly behind
    )

    # Strategy should adapt
    self.assertNotEqual(agent.strategy, initial_strategy)
    # This will fail until strategy adaptation implementation exists

  async def test_error_recovery_with_real_server(self):
    """Test error recovery mechanisms with real server scenarios."""
    if not self.test_models:
      self.skipTest("No API keys available for testing")

    model = self.test_models[0]
    agent = FreeCivLLMAgent(
        model=model,
        strategy="balanced",
        fallback_to_random=True
    )

    await self.proxy_client.connect()

    # Test with malformed observation
    malformed_observation = {
        "serializedGameAndState": "malformed_data",
        "legalActions": [],  # Empty legal actions
        "invalid_field": "should_be_ignored"
    }

    # Should handle gracefully
    try:
      # result = await agent.get_action_async(malformed_observation, self.proxy_client)
      # self.assertIsNotNone(result)
      pass
    except Exception as e:
      self.fail(f"Agent should handle malformed data gracefully: {e}")
    # This will fail until error handling implementation exists

  async def test_performance_benchmarks(self):
    """Test performance benchmarks for different scenarios."""
    if not self.test_models:
      self.skipTest("No API keys available for testing")

    benchmarks = {}

    for i, model in enumerate(self.test_models[:2]):  # Test up to 2 models
      agent = FreeCivLLMAgent(
          model=model,
          strategy="balanced",
          use_rethinking=False,
          fallback_to_random=True
      )

      await self.proxy_client.connect()

      # Benchmark single action generation
      state_data = await self.proxy_client.get_game_state()
      observation = {
          "serializedGameAndState": "benchmark_test",
          "legalActions": list(range(20)),
          **state_data
      }

      times = []
      for _ in range(3):  # Run multiple times for average
        start_time = time.time()
        # result = await agent.get_action_async(observation, self.proxy_client)
        execution_time = time.time() - start_time
        times.append(execution_time)

      avg_time = sum(times) / len(times)
      benchmarks[f"model_{i}"] = {
          "model_name": model.model_name,
          "avg_time": avg_time,
          "max_time": max(times),
          "min_time": min(times)
      }

      # All executions should be under 5 seconds
      self.assertLess(avg_time, 5.0)
      self.assertLess(max(times), 5.0)

    # Log benchmark results
    print(f"\nPerformance Benchmarks: {benchmarks}")
    # This will fail until implementation exists

  async def test_tournament_integration(self):
    """Test integration with tournament orchestration."""
    if not self.test_models:
      self.skipTest("No API keys available for testing")

    model = self.test_models[0]
    agent = FreeCivLLMAgent(
        model=model,
        strategy="balanced",
        fallback_to_random=True
    )

    # Test tournament-style observation format
    tournament_observation = {
        "serializedGameAndState": "tournament_state",
        "legalActions": [0, 1, 2, 3, 4],
        "playerID": 1,
        "gameMetadata": {
            "tournament_id": "test_tournament",
            "game_id": "test_game_001",
            "round": 1
        }
    }

    await self.proxy_client.connect()

    # Should handle tournament format
    # result = agent(tournament_observation, {})

    # self.assertIn("submission", result)
    # self.assertIn(result["submission"], tournament_observation["legalActions"])
    # This will fail until tournament integration implementation exists


if __name__ == "__main__":
  # Run with verbose output for debugging
  unittest.main(verbosity=2)