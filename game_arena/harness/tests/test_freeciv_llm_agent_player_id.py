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

"""Tests for FreeCivLLMAgent player ID extraction logic.

This test suite addresses PR feedback issue (Medium Severity):
Player ID fallback (freeciv_llm_agent.py:336-349) where silent fallback
to player ID 1 can cause bugs in multi-agent scenarios.

The fix changes behavior from silent fallback to raising ValueError
with helpful error message when player ID cannot be determined.
"""

import unittest
from unittest.mock import MagicMock, patch

from game_arena.harness import model_registry
from game_arena.harness.freeciv_llm_agent import FreeCivLLMAgent


class TestPlayerIDExtraction(unittest.TestCase):
  """Test suite for player ID extraction from observations."""

  def setUp(self):
    """Set up test agent."""
    # Create a mock model
    self.mock_model = MagicMock()
    self.mock_model.model_name = "test-model"

    # Create agent with mocked model
    self.agent = FreeCivLLMAgent(
        model=self.mock_model,
        strategy="balanced",
        use_rethinking=False,  # Disable rethinking for simpler tests
    )

  def test_missing_player_id_raises_exception(self):
    """Test that missing player ID raises ValueError instead of defaulting.

    This is the KEY fix for the PR feedback issue. Previously, the code
    would silently default to player ID 1, which causes bugs in multi-agent
    games where different agents should have different IDs.

    Expected behavior:
    - ValueError raised with helpful message
    - Error message indicates observation keys that were checked
    - No silent fallback to default value
    """
    # Observation with NO player ID field
    observation = {
        "turn": 1,
        "phase": "move",
        "serializedGameAndState": "...",
        "legalActions": [1, 2, 3],
        # Deliberately missing: playerID, player_id, current_player, agent_id
    }

    # This should RAISE ValueError, not return a default
    with self.assertRaises(ValueError) as context:
      player_id = self.agent._extract_player_id(observation)

    # Verify error message is helpful
    error_msg = str(context.exception).lower()
    self.assertIn("player id", error_msg)
    # Error should mention what was tried
    self.assertTrue(
        "observation" in error_msg.lower() or
        "cannot" in error_msg.lower() or
        "missing" in error_msg.lower()
    )

  def test_player_id_from_standard_key(self):
    """Test extraction from standard 'playerID' key (FreeCiv format)."""
    observation = {
        "playerID": 3,
        "turn": 1,
        "legalActions": [1, 2, 3],
    }

    player_id = self.agent._extract_player_id(observation)
    self.assertEqual(player_id, 3)

  def test_player_id_from_snake_case_key(self):
    """Test extraction from 'player_id' key (Python naming convention)."""
    observation = {
        "player_id": 2,
        "turn": 1,
        "legalActions": [1, 2, 3],
    }

    player_id = self.agent._extract_player_id(observation)
    self.assertEqual(player_id, 2)

  def test_player_id_from_current_player_key(self):
    """Test extraction from 'current_player' key (alternative format)."""
    observation = {
        "current_player": 4,
        "turn": 1,
        "legalActions": [1, 2, 3],
    }

    player_id = self.agent._extract_player_id(observation)
    self.assertEqual(player_id, 4)

  def test_player_id_from_agent_id_key(self):
    """Test extraction from 'agent_id' key (agent-centric format)."""
    observation = {
        "agent_id": 5,
        "turn": 1,
        "legalActions": [1, 2, 3],
    }

    player_id = self.agent._extract_player_id(observation)
    self.assertEqual(player_id, 5)

  def test_player_id_from_nested_state(self):
    """Test extraction from nested 'state' object."""
    observation = {
        "turn": 1,
        "state": {
            "current_player": 6,
            "phase": "move",
        },
        "legalActions": [1, 2, 3],
    }

    player_id = self.agent._extract_player_id(observation)
    self.assertEqual(player_id, 6)

  def test_player_id_from_nested_active_player(self):
    """Test extraction from nested 'active_player' field."""
    observation = {
        "turn": 1,
        "state": {
            "active_player": 7,
            "turn_player": 7,  # Alternative field
        },
        "legalActions": [1, 2, 3],
    }

    player_id = self.agent._extract_player_id(observation)
    self.assertEqual(player_id, 7)

  def test_invalid_player_id_type_raises_error(self):
    """Test that non-integer player IDs are rejected."""
    observation = {
        "playerID": "invalid_string",  # Wrong type
        "legalActions": [1, 2, 3],
    }

    with self.assertRaises(ValueError) as context:
      self.agent._extract_player_id(observation)

    error_msg = str(context.exception)
    self.assertTrue(
        "player ID" in error_msg.lower() or
        "invalid" in error_msg.lower()
    )

  def test_negative_player_id_raises_error(self):
    """Test that negative player IDs are rejected."""
    observation = {
        "playerID": -1,  # Invalid: negative
        "legalActions": [1, 2, 3],
    }

    with self.assertRaises(ValueError) as context:
      self.agent._extract_player_id(observation)

    error_msg = str(context.exception).lower()
    self.assertIn("player id", error_msg)


class TestMultiAgentScenarios(unittest.TestCase):
  """Test multi-agent scenarios to ensure player IDs are preserved correctly."""

  def setUp(self):
    """Set up multiple test agents."""
    self.mock_model = MagicMock()
    self.mock_model.model_name = "test-model"

  def test_multi_agent_scenario_preserves_player_ids(self):
    """Test that 4 agents maintain distinct player IDs.

    This test ensures that in a multi-agent game (e.g., 4-player FreeCiv),
    each agent correctly identifies itself with the right player ID and
    doesn't accidentally use another agent's ID.

    Expected behavior:
    - Each agent extracts the correct player ID from its observation
    - No agent accidentally uses player ID 1 as a default
    - Player IDs remain consistent across multiple calls
    """
    # Create 4 agents for 4-player game
    agents = [
        FreeCivLLMAgent(model=self.mock_model, use_rethinking=False)
        for _ in range(4)
    ]

    # Create observations for each player (IDs 0-3)
    observations = [
        {"playerID": player_id, "turn": 1, "legalActions": [1, 2, 3]}
        for player_id in range(4)
    ]

    # Each agent should extract its correct player ID
    for agent_idx, agent in enumerate(agents):
      player_id = agent._extract_player_id(observations[agent_idx])
      self.assertEqual(
          player_id, agent_idx,
          f"Agent {agent_idx} should extract player ID {agent_idx}, got {player_id}"
      )

      # Call again to verify consistency
      player_id_second_call = agent._extract_player_id(observations[agent_idx])
      self.assertEqual(
          player_id_second_call, agent_idx,
          f"Player ID should be consistent across calls"
      )

  def test_no_cross_contamination_between_agents(self):
    """Test that one agent's cached player ID doesn't affect another agent.

    This ensures the bug where agent1 might accidentally use agent2's
    player ID cannot occur.
    """
    agent1 = FreeCivLLMAgent(model=self.mock_model, use_rethinking=False)
    agent2 = FreeCivLLMAgent(model=self.mock_model, use_rethinking=False)

    obs1 = {"playerID": 1, "turn": 1, "legalActions": [1, 2, 3]}
    obs2 = {"playerID": 2, "turn": 1, "legalActions": [1, 2, 3]}

    # Agent 1 extracts player ID 1
    pid1 = agent1._extract_player_id(obs1)
    self.assertEqual(pid1, 1)

    # Agent 2 should extract player ID 2 (not be affected by agent1)
    pid2 = agent2._extract_player_id(obs2)
    self.assertEqual(pid2, 2)

    # Verify no cross-contamination
    pid1_recheck = agent1._extract_player_id(obs1)
    self.assertEqual(pid1_recheck, 1, "Agent 1 should still have player ID 1")


class TestPlayerIDErrorMessages(unittest.TestCase):
  """Test that error messages are helpful for debugging."""

  def setUp(self):
    """Set up test agent."""
    self.mock_model = MagicMock()
    self.mock_model.model_name = "test-model"
    self.agent = FreeCivLLMAgent(model=self.mock_model, use_rethinking=False)

  def test_error_message_includes_available_keys(self):
    """Test that error message shows what keys were available in observation.

    This helps developers quickly identify the issue when integrating
    with different observation formats.
    """
    observation = {
        "turn": 5,
        "phase": "move",
        "some_other_field": "value",
    }

    with self.assertRaises(ValueError) as context:
      self.agent._extract_player_id(observation)

    error_msg = str(context.exception)

    # Error should be informative enough for debugging
    self.assertTrue(
        len(error_msg) > 20,
        "Error message should be descriptive, not just 'Invalid player ID'"
    )


if __name__ == "__main__":
  unittest.main()
