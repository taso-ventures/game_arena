"""Tests for FreeCiv prompt builder."""

import unittest
from unittest.mock import MagicMock, patch
import os

from game_arena.harness.prompts.freeciv_prompts import FreeCivPromptBuilder


class FreeCivPromptPlayerConversionTest(unittest.TestCase):
  """Tests for _prepare_observation() player list to dict conversion."""

  def setUp(self):
    """Set up test fixtures."""
    # Create prompt builder (uses default config path)
    self.prompt_builder = FreeCivPromptBuilder()

  def test_prepare_observation_converts_players_list_to_dict(self):
    """Test that players list is converted to dict keyed by player_id."""
    observation = {
        "turn": 5,
        "phase": "move",
        "players": [
            {"id": 0, "name": "Player1", "score": 100},
            {"id": 1, "name": "Player2", "score": 150},
            {"id": 2, "name": "Player3", "score": 75},
        ],
        "units": [],
        "cities": [],
    }

    result = self.prompt_builder._prepare_observation(observation)

    # Players should be converted to dict
    self.assertIsInstance(result["players"], dict)
    self.assertEqual(len(result["players"]), 3)

    # Check all players are accessible by ID
    self.assertIn(0, result["players"])
    self.assertIn(1, result["players"])
    self.assertIn(2, result["players"])

    # Verify player data is preserved
    self.assertEqual(result["players"][0]["name"], "Player1")
    self.assertEqual(result["players"][0]["score"], 100)
    self.assertEqual(result["players"][1]["name"], "Player2")
    self.assertEqual(result["players"][1]["score"], 150)

  def test_prepare_observation_handles_player_id_field_variations(self):
    """Test that different player ID field names are handled."""
    # Test with 'player_id' field
    observation_with_player_id = {
        "players": [
            {"player_id": 0, "name": "Player1"},
            {"player_id": 1, "name": "Player2"},
        ]
    }
    result = self.prompt_builder._prepare_observation(observation_with_player_id)
    self.assertIn(0, result["players"])
    self.assertIn(1, result["players"])

    # Test with 'playerno' field
    observation_with_playerno = {
        "players": [
            {"playerno": 0, "name": "Player1"},
            {"playerno": 1, "name": "Player2"},
        ]
    }
    result = self.prompt_builder._prepare_observation(observation_with_playerno)
    self.assertIn(0, result["players"])
    self.assertIn(1, result["players"])

  def test_prepare_observation_preserves_players_dict(self):
    """Test that players dict is preserved without modification."""
    observation = {
        "turn": 5,
        "players": {
            0: {"id": 0, "name": "Player1", "score": 100},
            1: {"id": 1, "name": "Player2", "score": 150},
        },
        "units": [],
    }

    result = self.prompt_builder._prepare_observation(observation)

    # Players should remain as dict
    self.assertIsInstance(result["players"], dict)
    self.assertEqual(len(result["players"]), 2)
    self.assertEqual(result["players"][0]["name"], "Player1")
    self.assertEqual(result["players"][1]["name"], "Player2")

  def test_prepare_observation_handles_missing_players(self):
    """Test that observations without players field are handled."""
    observation = {
        "turn": 5,
        "phase": "move",
        "units": [],
        "cities": [],
    }

    result = self.prompt_builder._prepare_observation(observation)

    # Should not crash, and should not add players field
    self.assertNotIn("players", result)
    self.assertEqual(result["turn"], 5)
    self.assertEqual(result["phase"], "move")

  def test_prepare_observation_handles_invalid_players_type(self):
    """Test that invalid players types are replaced with empty dict."""
    # Test with string
    observation_string = {
        "turn": 5,
        "players": "invalid_string",
    }

    with patch('logging.warning') as mock_warning:
      result = self.prompt_builder._prepare_observation(observation_string)

      # Should replace with empty dict and log warning
      self.assertIsInstance(result["players"], dict)
      self.assertEqual(len(result["players"]), 0)
      mock_warning.assert_called()

    # Test with number
    observation_number = {
        "turn": 5,
        "players": 42,
    }

    with patch('logging.warning') as mock_warning:
      result = self.prompt_builder._prepare_observation(observation_number)

      self.assertIsInstance(result["players"], dict)
      self.assertEqual(len(result["players"]), 0)
      mock_warning.assert_called()

  def test_prepare_observation_handles_empty_players_list(self):
    """Test that empty players list is converted to empty dict."""
    observation = {
        "turn": 5,
        "players": [],
    }

    result = self.prompt_builder._prepare_observation(observation)

    # Should be empty dict
    self.assertIsInstance(result["players"], dict)
    self.assertEqual(len(result["players"]), 0)

  def test_prepare_observation_skips_invalid_player_entries(self):
    """Test that invalid player entries in list are skipped."""
    observation = {
        "players": [
            {"id": 0, "name": "Player1"},  # Valid
            "invalid_string",  # Invalid - not a dict
            {"name": "NoID"},  # Invalid - no ID field
            {"id": 2, "name": "Player3"},  # Valid
        ]
    }

    result = self.prompt_builder._prepare_observation(observation)

    # Should only include valid players
    self.assertIsInstance(result["players"], dict)
    self.assertEqual(len(result["players"]), 2)
    self.assertIn(0, result["players"])
    self.assertIn(2, result["players"])
    self.assertNotIn(1, result["players"])

  def test_prepare_observation_preserves_other_fields(self):
    """Test that other observation fields are preserved during conversion."""
    observation = {
        "turn": 10,
        "phase": "combat",
        "players": [{"id": 0, "name": "Player1"}],
        "units": [{"id": 1, "type": "warrior"}],
        "cities": [{"id": 1, "name": "Rome"}],
        "map": {"width": 50, "height": 50},
    }

    result = self.prompt_builder._prepare_observation(observation)

    # All other fields should be preserved
    self.assertEqual(result["turn"], 10)
    self.assertEqual(result["phase"], "combat")
    self.assertEqual(result["units"], [{"id": 1, "type": "warrior"}])
    self.assertEqual(result["cities"], [{"id": 1, "name": "Rome"}])
    self.assertEqual(result["map"], {"width": 50, "height": 50})

  def test_compress_observation_handles_integer_unit_types(self):
    """Test that integer unit types from proxy don't crash compression."""
    obs = {
        "turn": 5,
        "units": [
            {"id": 1, "type": 56, "owner": 0, "x": 10, "y": 10, "hp": 10},  # Integer type ID
            {"id": 2, "type": "warrior", "owner": 0, "x": 11, "y": 10, "hp": 10},  # String
            {"id": 3, "type": None, "owner": 0, "x": 12, "y": 10, "hp": 10},  # None
        ]
    }
    model_config = {"max_tokens": 4000}

    # Should not crash with AttributeError
    compressed = self.prompt_builder.context_manager.compress_observation(obs, 4000)

    # Verify result is valid
    self.assertIsInstance(compressed, dict)
    self.assertIn("units", compressed)

  def test_analyze_victory_conditions_handles_integer_types(self):
    """Test victory analysis with integer unit types."""
    obs = {
        "turn": 10,
        "cities": [{"id": 1, "population": 5}],
        "units": [
            {"id": 1, "type": 3, "owner": 0},  # Integer type ID
            {"id": 2, "type": "warrior", "owner": 0},  # String
            {"id": 3, "type": None, "owner": 0},  # None
        ]
    }

    # Should not crash with AttributeError
    victory_type, progress = self.prompt_builder._analyze_victory_conditions(obs)

    # Verify result types
    self.assertIsInstance(victory_type, str)
    self.assertIsInstance(progress, int)
    self.assertGreaterEqual(progress, 0)
    self.assertLessEqual(progress, 100)


if __name__ == "__main__":
  unittest.main()
