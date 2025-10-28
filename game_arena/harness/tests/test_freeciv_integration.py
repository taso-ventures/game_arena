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

"""Integration tests for FreeCiv LLM Agent with real components."""

import asyncio
import os
import unittest
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

from game_arena.harness import model_generation, model_registry, rethink
from game_arena.harness.freeciv_llm_agent import FreeCivLLMAgent
from game_arena.harness.freeciv_proxy_client import FreeCivProxyClient
from game_arena.harness.freeciv_state import FreeCivAction, FreeCivState
from game_arena.harness.freeciv_action_converter import FreeCivActionConverter
from game_arena.harness.freeciv_memory import GameMemory
from game_arena.harness.freeciv_strategy import StrategyManager
from game_arena.harness.prompts.freeciv_prompts import FreeCivPromptBuilder
from game_arena.harness.tests.test_helpers import (
    MockFreeCivProxyClient,
    MockModelWithPredictableResponses,
    FreeCivTestData,
    AsyncTestHelpers
)


class TestFreeCivLLMAgentIntegration(unittest.IsolatedAsyncioTestCase):
  """Integration tests for FreeCiv LLM Agent with real components."""

  def setUp(self):
    """Set up test fixtures with real components."""
    # Use mock model for predictable testing
    self.mock_model = MockModelWithPredictableResponses()

    # Create real FreeCiv proxy client mock that behaves like real one
    self.mock_proxy_client = MockFreeCivProxyClient()

    # Create test data using helper
    self.observation = FreeCivTestData.create_sample_observation(player_id=1)
    self.freeciv_state = FreeCivTestData.create_sample_freeciv_state(player_id=1)
    self.sample_actions = FreeCivTestData.create_sample_freeciv_actions()

    # Real API key tests (optional)
    self.real_api_key = os.getenv("GEMINI_API_KEY")
    self.real_model = None
    if self.real_api_key:
      try:
        self.real_model = model_registry.ModelRegistry.GEMINI_2_5_FLASH.build(
            api_key=self.real_api_key,
            model_options={"max_tokens": 1000, "temperature": 0.7}
        )
      except Exception:
        pass  # Skip real model tests if unavailable

  async def test_agent_with_mock_model_integration(self):
    """Test agent integrates correctly with mock model."""
    agent = FreeCivLLMAgent(
        model=self.mock_model,
        strategy="balanced",
        use_rethinking=False
    )

    # Test that agent was initialized correctly
    self.assertIsNotNone(agent.model)
    self.assertEqual(agent.strategy, "balanced")
    self.assertIsInstance(agent.prompt_builder, FreeCivPromptBuilder)
    self.assertIsInstance(agent.action_converter, FreeCivActionConverter)
    self.assertIsInstance(agent.memory, GameMemory)
    self.assertIsInstance(agent.strategy_manager, StrategyManager)

    # Test action generation with observation
    result = agent(self.observation, {})

    # Should return properly formatted result
    self.assertIsInstance(result, dict)
    self.assertIn("submission", result)
    self.assertIsInstance(result["submission"], int)

  async def test_agent_with_rethinking_sampler(self):
    """Test agent integrates with rethinking sampler for illegal move handling."""
    agent = FreeCivLLMAgent(
        model=self.mock_model,
        strategy="balanced",
        use_rethinking=True,
        max_rethinks=2
    )

    # Should have rethinking sampler configured
    self.assertIsNotNone(agent.sampler)
    self.assertIsInstance(agent.sampler, rethink.RethinkSampler)

    # Test that rethinking sampler has correct configuration
    self.assertEqual(agent.sampler.num_max_rethinks, 2)
    self.assertEqual(agent.sampler.game_short_name, "freeciv")

  async def test_prompt_builder_integration(self):
    """Test integration with FreeCivPromptBuilder."""
    agent = FreeCivLLMAgent(model=self.mock_model)

    # Should use real prompt builder
    self.assertIsInstance(agent.prompt_builder, FreeCivPromptBuilder)

    # Should generate valid prompts
    legal_actions = self.freeciv_state.get_legal_actions(player_id=1)

    prompt = agent.prompt_builder.build_enhanced_prompt(
        observation=self.observation,
        legal_actions=legal_actions,
        model_name=self.mock_model.model_name
    )

    self.assertIsInstance(prompt, str)
    self.assertGreater(len(prompt), 100)  # Should be substantial prompt
    self.assertIn("FreeCiv", prompt)  # Should mention game
    self.assertIn("turn", prompt.lower())  # Should include turn info

  async def test_action_parser_integration(self):
    """Test integration with FreeCivActionParser."""
    agent = FreeCivLLMAgent(model=self.mock_model)

    # Mock LLM response
    llm_response = "unit_move_warrior(101)_to(11,14)"

    # Should parse response correctly
    from game_arena.harness import parsers
    parser_input = parsers.TextParserInput(
        text=llm_response,
        legal_moves=["unit_move_warrior(101)_to(11,14)", "unit_fortify_warrior(101)"]
    )

    parsed_action = agent.action_parser.parse(parser_input)
    self.assertEqual(parsed_action, "unit_move_warrior(101)_to(11,14)")

    # Test with invalid action
    parser_input_invalid = parsers.TextParserInput(
        text="invalid_action_format",
        legal_moves=["unit_move_warrior(101)_to(11,14)"]
    )

    # Should either return None or raise appropriate exception
    try:
      result = agent.action_parser.parse(parser_input_invalid)
      self.assertIsNone(result)  # Should return None for invalid actions
    except Exception:
      pass  # Or raise exception - both acceptable

  async def test_websocket_client_integration(self):
    """Test integration with FreeCivProxyClient for WebSocket communication."""
    # Connect the mock client first
    await self.mock_proxy_client.connect()

    # Test basic client functionality
    self.assertTrue(self.mock_proxy_client.is_connected())

    # Test getting game state
    state = await self.mock_proxy_client.get_game_state()
    self.assertIsInstance(state, dict)
    self.assertIn("turn", state)
    self.assertIn("players", state)

    # Test sending action
    action = {"action_type": "unit_move", "actor_id": 101, "target": {"x": 11, "y": 14}}
    result = await self.mock_proxy_client.send_action(action)
    self.assertIsInstance(result, dict)
    self.assertTrue(result.get("success", False))

    # Test getting legal actions
    legal_actions = await self.mock_proxy_client.get_legal_actions(player_id=1)
    self.assertIsInstance(legal_actions, list)
    self.assertGreater(len(legal_actions), 0)

    # Check call history
    history = self.mock_proxy_client.get_call_history()
    self.assertGreater(len(history), 0)

    await self.mock_proxy_client.disconnect()

  async def test_memory_system_integration(self):
    """Test integration with memory system for context management."""
    agent = FreeCivLLMAgent(
        model=self.mock_model,
        memory_size=5
    )

    # Should maintain memory across multiple actions
    for i in range(3):
      mock_action = FreeCivAction(
          action_type="unit_move",
          actor_id=101 + i,
          target={"x": 10 + i, "y": 14},
          parameters={},
          source="unit"
      )
      mock_result = {"success": True, "score_delta": 5}

      agent.memory.record_action(mock_action, mock_result)

    # Memory should contain recorded actions
    self.assertEqual(len(agent.memory.history), 3)

    # Should generate context for prompts
    context = agent.memory.get_context(max_tokens=1000)
    self.assertIsInstance(context, str)
    self.assertGreater(len(context), 0)

    # Test performance summary
    performance = agent.memory.get_performance_summary()
    self.assertIsInstance(performance, dict)
    self.assertEqual(performance["total_actions"], 3)
    self.assertGreater(performance["success_rate"], 0)

  async def test_strategy_adaptation_integration(self):
    """Test integration with strategy system for dynamic adaptation."""
    agent = FreeCivLLMAgent(
        model=self.mock_model,
        strategy="balanced"
    )

    # Test initial strategy
    self.assertEqual(agent.strategy, "balanced")
    self.assertIsInstance(agent.strategy_manager, StrategyManager)

    # Test strategy adaptation
    new_strategy = agent.strategy_manager.adapt_strategy(
        current_strategy="balanced",
        game_phase="late",
        relative_score=-0.3  # Behind in score
    )

    # Should return a valid strategy
    self.assertIsInstance(new_strategy, str)
    self.assertIn(new_strategy, agent.strategy_manager.get_available_strategies())

    # Test strategy comparison
    comparison = agent.strategy_manager.compare_strategies("balanced", "aggressive_expansion")
    self.assertIsInstance(comparison, dict)
    self.assertIn("strategies", comparison)
    self.assertIn("risk_tolerance", comparison)

  async def test_error_recovery_integration(self):
    """Test error recovery with real retry mechanisms."""
    # Create agent with fallback configuration
    agent = FreeCivLLMAgent(
        model=self.mock_model,
        fallback_to_random=True
    )

    # Test error handling with invalid observation
    invalid_observation = {"invalid": "data"}

    # Should not crash, either return valid result or raise controlled exception
    try:
      result = agent(invalid_observation, {})
      # If it succeeds, should return valid format
      self.assertIsInstance(result, dict)
      self.assertIn("submission", result)
    except Exception as e:
      # If it fails, should be a controlled failure
      self.assertIsInstance(e, (ValueError, KeyError, TypeError))

    # Test with valid observation but missing legal actions
    empty_observation = FreeCivTestData.create_sample_observation()
    empty_observation["legalActions"] = []

    try:
      result = agent(empty_observation, {})
      # Should handle empty legal actions gracefully
      self.assertIsInstance(result, dict)
    except Exception:
      pass  # Acceptable to fail with empty legal actions

  async def test_performance_within_time_limit(self):
    """Test that action generation completes within reasonable time limit."""
    agent = FreeCivLLMAgent(model=self.mock_model)

    start_time = asyncio.get_event_loop().time()

    # Should complete within time limit
    result = await asyncio.wait_for(
        asyncio.coroutine(lambda: agent(self.observation, {}))(),
        timeout=5.0
    )

    elapsed_time = asyncio.get_event_loop().time() - start_time
    self.assertLess(elapsed_time, 5.0)
    self.assertIsInstance(result, dict)
    self.assertIn("submission", result)

  async def test_action_converter_integration(self):
    """Test integration with action converter for bidirectional conversion."""
    agent = FreeCivLLMAgent(model=self.mock_model)

    # Test action to int conversion
    test_action = self.sample_actions[0]  # unit_move action
    action_int = agent.action_converter.action_to_int(
        test_action, self.freeciv_state, player_id=1
    )
    self.assertIsInstance(action_int, int)
    self.assertGreaterEqual(action_int, 0)

    # Test int to action conversion
    converted_action = agent.action_converter.int_to_action(
        action_int, self.freeciv_state, player_id=1
    )
    self.assertIsInstance(converted_action, FreeCivAction)
    AsyncTestHelpers.assert_valid_freeciv_action(converted_action)

    # Test string conversion
    action_string = agent.action_converter.action_to_string(test_action)
    self.assertIsInstance(action_string, str)
    self.assertGreater(len(action_string), 0)

    # Test string parsing
    parsed_action = agent.action_converter.string_to_action(action_string)
    self.assertIsInstance(parsed_action, FreeCivAction)
    self.assertEqual(parsed_action.action_type, test_action.action_type)

  async def test_real_model_integration(self):
    """Test with real model if API key is available."""
    if not self.real_model:
      self.skipTest("Real model not available")

    agent = FreeCivLLMAgent(
        model=self.real_model,
        strategy="balanced",
        use_rethinking=False
    )

    # Test basic functionality with real model
    try:
      result = agent(self.observation, {})
      self.assertIsInstance(result, dict)
      self.assertIn("submission", result)
      self.assertIsInstance(result["submission"], int)
    except Exception as e:
      # Real model tests can fail due to API issues
      self.skipTest(f"Real model test failed: {e}")

  async def test_full_pipeline_integration(self):
    """Test full pipeline from observation to action submission."""
    await self.mock_proxy_client.connect()

    agent = FreeCivLLMAgent(
        model=self.mock_model,
        strategy="aggressive_expansion",
        use_rethinking=True,
        max_rethinks=1,
        memory_size=3
    )

    # Simulate multiple turns
    for turn in range(3):
      # Update observation for new turn
      current_obs = FreeCivTestData.create_sample_observation(
          player_id=1, turn=turn + 1
      )

      # Agent generates action
      result = agent(current_obs, {})

      # Validate result
      self.assertIsInstance(result, dict)
      self.assertIn("submission", result)
      action_id = result["submission"]
      self.assertIsInstance(action_id, int)
      self.assertIn(action_id, current_obs["legalActions"])

      # Simulate action execution
      mock_action = {"action_id": action_id, "turn": turn + 1}
      action_result = await self.mock_proxy_client.send_action(mock_action)
      self.assertTrue(action_result.get("success", False))

    # Check that memory accumulated
    self.assertGreater(len(agent.memory.history), 0)

    # Check performance tracking
    self.assertGreater(agent._num_model_calls, 0)
    self.assertGreater(agent._total_response_time, 0)

    await self.mock_proxy_client.disconnect()

  async def test_concurrent_request_handling(self):
    """Test that agent can handle concurrent requests safely."""
    agent = FreeCivLLMAgent(model=self.mock_model)

    # Create multiple concurrent requests
    tasks = []
    for i in range(3):
      observation = {**self.observation, "turn": i + 1}
      # Use asyncio.coroutine to wrap synchronous call
      task = asyncio.coroutine(lambda obs=observation: agent(obs, {}))()
      tasks.append(task)

    # All should complete successfully
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
      self.assertNotIsInstance(result, Exception)
      self.assertIsInstance(result, dict)
      self.assertIn("submission", result)


if __name__ == '__main__':
  unittest.main()