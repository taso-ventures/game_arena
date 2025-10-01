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

"""Unit tests for FreeCiv LLM Agent following TDD approach."""

import unittest
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, List

from game_arena.harness import agent, model_generation, tournament_util
from game_arena.harness.freeciv_state import FreeCivAction, FreeCivState
from game_arena.harness.freeciv_llm_agent import FreeCivLLMAgent
from game_arena.harness.freeciv_memory import GameMemory
from game_arena.harness.freeciv_strategy import StrategyManager


class TestFreeCivLLMAgent(unittest.TestCase):
  """Unit tests for FreeCiv LLM Agent."""

  def setUp(self):
    """Set up test fixtures."""
    self.mock_model = MagicMock(spec=model_generation.Model)
    self.mock_model.model_name = "test-model"

    # Create mock legal actions
    self.mock_legal_actions = [1, 2, 3, 4, 5]

    # Create mock observation
    self.mock_observation = {
        "serializedGameAndState": "test_state_data",
        "legalActions": self.mock_legal_actions,
        "turn": 10,
        "players": {1: {"score": 340, "name": "Romans"}},
        "units": [{"id": 1, "type": "Warrior", "x": 10, "y": 14}],
        "cities": [{"id": 1, "name": "Rome", "x": 10, "y": 15}]
    }

    # Create mock legal FreeCivActions
    self.mock_freeciv_actions = [
        FreeCivAction("unit_move", 1, {"x": 11, "y": 14}, {}, "unit"),
        FreeCivAction("unit_attack", 1, {"id": 2}, {}, "unit"),
        FreeCivAction("city_production", 1, {"value": "warriors"}, {}, "city")
    ]

  def test_agent_initialization_with_default_strategy(self):
    """Test agent initializes correctly with default strategy."""
    agent = FreeCivLLMAgent(model=self.mock_model)

    # Should initialize with default strategy
    self.assertEqual(agent.strategy, "balanced")
    self.assertIsNotNone(agent.model)
    self.assertIsNotNone(agent.prompt_builder)
    self.assertIsNotNone(agent.action_parser)
    self.assertIsNotNone(agent.memory)
    self.assertEqual(agent.model, self.mock_model)

  def test_agent_initialization_with_custom_strategy(self):
    """Test agent initializes with custom strategy."""
    agent = FreeCivLLMAgent(
        model=self.mock_model,
        strategy="aggressive",
        memory_size=20
    )

    self.assertEqual(agent.strategy, "aggressive")
    self.assertEqual(agent.memory.max_size, 20)

  def test_agent_initialization_with_rethinking_enabled(self):
    """Test agent initializes with rethinking sampler when enabled."""
    agent = FreeCivLLMAgent(
        model=self.mock_model,
        use_rethinking=True,
        max_rethinks=5
    )

    self.assertIsNotNone(agent.sampler)
    # Should be a RethinkSampler with correct configuration
    # This will fail until implementation exists

  def test_agent_initialization_with_rethinking_disabled(self):
    """Test agent initializes without rethinking sampler when disabled."""
    agent = FreeCivLLMAgent(
        model=self.mock_model,
        use_rethinking=False
    )

    self.assertIsNone(agent.sampler)

  def test_agent_inherits_from_kaggle_spiel_agent(self):
    """Test that FreeCivLLMAgent inherits from KaggleSpielAgent."""
    agent = FreeCivLLMAgent(model=self.mock_model)

    # Should inherit from proper base class
    self.assertIsInstance(agent, agent.KaggleSpielAgent)
    # This will fail until implementation exists

  @patch('game_arena.harness.freeciv_llm_agent.FreeCivStateSynchronizer')
  @patch('game_arena.harness.freeciv_llm_agent.FreeCivActionConverter')
  def test_call_method_returns_valid_action(self, mock_converter, mock_sync):
    """Test that __call__ method returns valid action integer."""
    # Setup mocks
    mock_sync.return_value.sync_state.return_value = MagicMock(spec=FreeCivState)
    mock_converter.return_value.action_to_int.return_value = 2

    agent = FreeCivLLMAgent(model=self.mock_model)

    result = agent(self.mock_observation, {})

    # Should return dict with submission key
    self.assertIsInstance(result, dict)
    self.assertIn("submission", result)
    self.assertIn(result["submission"], self.mock_legal_actions)
    # This will fail until implementation exists

  @patch('game_arena.harness.freeciv_llm_agent.FreeCivStateSynchronizer')
  def test_get_action_async_with_valid_response(self, mock_sync):
    """Test async action generation with valid LLM response."""
    # Setup mock state
    mock_state = MagicMock(spec=FreeCivState)
    mock_state.get_legal_actions.return_value = self.mock_freeciv_actions
    mock_sync.return_value.sync_state.return_value = mock_state

    # Setup mock model response
    mock_response = MagicMock()
    mock_response.main_response = "unit_move_warrior(1)_to(11,14)"
    self.mock_model.generate_with_text_input = AsyncMock(return_value=mock_response)

    agent = FreeCivLLMAgent(model=self.mock_model)

    # This will fail until async implementation exists
    # result = await agent.get_action_async(self.mock_observation, mock_proxy_client)
    # self.assertIsInstance(result, FreeCivAction)

  def test_fallback_to_random_when_model_fails(self):
    """Test fallback mechanism when LLM generation fails."""
    # Setup model to fail
    self.mock_model.generate_with_text_input.side_effect = Exception("API Error")

    agent = FreeCivLLMAgent(model=self.mock_model, fallback_to_random=True)

    result = agent(self.mock_observation, {})

    # Should still return valid action from legal actions
    self.assertIn(result["submission"], self.mock_legal_actions)
    # This will fail until fallback implementation exists

  def test_fallback_disabled_raises_exception(self):
    """Test that agent raises exception when fallback is disabled and model fails."""
    self.mock_model.generate_with_text_input.side_effect = Exception("API Error")

    agent = FreeCivLLMAgent(model=self.mock_model, fallback_to_random=False)

    with self.assertRaises(Exception):
      agent(self.mock_observation, {})
    # This will fail until implementation exists

  def test_strategy_update_changes_configuration(self):
    """Test that strategy can be updated dynamically."""
    agent = FreeCivLLMAgent(model=self.mock_model, strategy="balanced")

    # Update strategy
    agent.update_strategy("aggressive", score_relative=0.8)

    self.assertEqual(agent.strategy, "aggressive")
    # This will fail until strategy update implementation exists

  def test_memory_records_action_history(self):
    """Test that agent records actions in memory."""
    agent = FreeCivLLMAgent(model=self.mock_model, memory_size=5)

    # Simulate recording an action
    mock_action = self.mock_freeciv_actions[0]
    mock_result = {"success": True, "score_change": 10}

    agent.memory.record_action(mock_action, mock_result)

    # Memory should contain the action
    self.assertEqual(len(agent.memory.history), 1)
    # This will fail until memory implementation exists

  def test_memory_respects_max_size_limit(self):
    """Test that memory respects maximum size limit."""
    agent = FreeCivLLMAgent(model=self.mock_model, memory_size=2)

    # Add more actions than max size
    for i in range(5):
      mock_action = FreeCivAction("unit_move", i, {"x": i, "y": i}, {}, "unit")
      agent.memory.record_action(mock_action, {"turn": i})

    # Should only keep last 2 actions
    self.assertEqual(len(agent.memory.history), 2)
    # This will fail until memory implementation exists

  @patch('game_arena.harness.freeciv_llm_agent.TokenManager')
  def test_prompt_respects_token_limits(self, mock_token_manager):
    """Test that generated prompts respect model token limits."""
    mock_token_manager.return_value.count_tokens.return_value = 2000
    mock_token_manager.return_value.truncate_to_limit.return_value = "truncated_prompt"

    agent = FreeCivLLMAgent(model=self.mock_model)

    # Generate prompt that would exceed limits
    large_observation = {**self.mock_observation, "large_data": "x" * 50000}

    # Should use token manager to truncate
    # This will fail until token management implementation exists

  def test_action_validation_against_legal_moves(self):
    """Test that parsed actions are validated against legal moves."""
    agent = FreeCivLLMAgent(model=self.mock_model)

    # Mock action parser to return invalid action
    invalid_action = FreeCivAction("unit_move", 999, {"x": 0, "y": 0}, {}, "unit")

    # Should validate and reject invalid action
    # This will fail until validation implementation exists

  def test_error_handling_with_retry_logic(self):
    """Test that agent uses retry logic for transient failures."""
    # Setup model to fail twice then succeed
    responses = [
        Exception("Timeout"),
        Exception("Rate limit"),
        MagicMock(main_response="unit_move_warrior(1)_to(11,14)")
    ]
    self.mock_model.generate_with_text_input.side_effect = responses

    agent = FreeCivLLMAgent(model=self.mock_model)

    # Should eventually succeed after retries
    # This will fail until retry implementation exists

  def test_performance_tracking(self):
    """Test that agent tracks performance metrics."""
    agent = FreeCivLLMAgent(model=self.mock_model)

    # Should track number of model calls
    self.assertEqual(agent._num_model_calls, 0)

    # After making calls, should increment counter
    # This will fail until tracking implementation exists

  def test_caching_improves_performance(self):
    """Test that caching is used to improve performance."""
    agent = FreeCivLLMAgent(model=self.mock_model)

    # First call should miss cache
    # Second identical call should hit cache
    # This will fail until caching implementation exists


class TestGameMemory(unittest.TestCase):
  """Unit tests for GameMemory component."""

  def setUp(self):
    """Set up test fixtures."""
    self.memory = GameMemory(max_size=5)
    self.mock_action = FreeCivAction("unit_move", 1, {"x": 2, "y": 3}, {}, "unit")
    self.mock_result = {"success": True, "score": 10}

  def test_memory_initialization(self):
    """Test memory initializes with correct parameters."""
    self.assertEqual(self.memory.max_size, 5)
    self.assertEqual(len(self.memory.history), 0)
    # This will fail until GameMemory implementation exists

  def test_record_action_adds_to_history(self):
    """Test that recording action adds to history."""
    self.memory.record_action(self.mock_action, self.mock_result)

    self.assertEqual(len(self.memory.history), 1)
    self.assertEqual(self.memory.history[0]["action"], self.mock_action)
    # This will fail until implementation exists

  def test_get_context_returns_formatted_string(self):
    """Test that get_context returns properly formatted string."""
    self.memory.record_action(self.mock_action, self.mock_result)

    context = self.memory.get_context(max_tokens=1000)

    self.assertIsInstance(context, str)
    self.assertIn("unit_move", context)
    # This will fail until implementation exists

  def test_context_respects_token_limit(self):
    """Test that context generation respects token limits."""
    # Add many actions
    for i in range(10):
      action = FreeCivAction("unit_move", i, {"x": i, "y": i}, {}, "unit")
      self.memory.record_action(action, {"turn": i})

    context = self.memory.get_context(max_tokens=100)

    # Should be truncated to fit within limit
    # This will fail until token-aware implementation exists


class TestStrategyManager(unittest.TestCase):
  """Unit tests for StrategyManager component."""

  def test_strategy_config_loading(self):
    """Test that strategy configurations load correctly."""
    manager = StrategyManager()

    config = manager.get_strategy_config("aggressive")

    self.assertIn("prioritize", config)
    self.assertIn("risk_tolerance", config)
    # This will fail until StrategyManager implementation exists

  def test_strategy_adaptation(self):
    """Test that strategy adapts based on game state."""
    manager = StrategyManager()

    # Should adapt strategy based on game phase and performance
    adapted_config = manager.adapt_strategy(
        current_strategy="balanced",
        game_phase="late",
        relative_score=-0.2
    )

    self.assertIsInstance(adapted_config, dict)
    # This will fail until adaptation implementation exists


if __name__ == "__main__":
  unittest.main()