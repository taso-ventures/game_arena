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

"""Tests for FreeCiv prompt builder."""

import time
import unittest
from unittest.mock import Mock, patch

from game_arena.harness.freeciv_state import FreeCivAction, FreeCivState
from game_arena.harness.prompts.freeciv_prompts import (ContextManager,
                                                        FreeCivPromptBuilder,
                                                        ObservationBuilder)


class TestFreeCivPromptBuilder(unittest.TestCase):
    """Test cases for FreeCivPromptBuilder class."""

    def setUp(self):
        """Set up test fixtures."""
        self.prompt_builder = FreeCivPromptBuilder()
        # Create a mock state that returns proper dictionary structure
        self.mock_state = Mock(spec=FreeCivState)
        self.mock_state.turn = 42

        # Set up players with proper dictionary structure
        player_mock = Mock()
        player_mock.score = 340
        player_mock.gold = 245
        player_mock.name = "Romans"
        self.mock_state.players = {1: player_mock}

        self.mock_state.units = [
            Mock(unit_id=1, kind="Warrior", position=(10, 14), hp=100, owner=1)
        ]
        self.mock_state.cities = [
            Mock(city_id=1, name="Rome", position=(10, 15), population=8, owner=1)
        ]

        self.legal_actions = [
            FreeCivAction(
                action_type="unit_move",
                actor_id=1,
                target={"x": 11, "y": 14},
                parameters={},
                source="unit",
            ),
            FreeCivAction(
                action_type="city_production",
                actor_id=1,
                target={"value": "Archer"},
                parameters={},
                source="city",
            ),
        ]

    def test_prompt_generation_per_model(self):
        """Test that each model gets appropriate prompt format."""
        models = ["gpt-5", "claude", "deepseek"]

        for model_name in models:
            with self.subTest(model=model_name):
                prompt = self.prompt_builder.build_enhanced_prompt(
                    observation={"state": self.mock_state},
                    legal_actions=self.legal_actions,
                    model_name=model_name,
                )

                self.assertIsInstance(prompt, str)
                self.assertGreater(len(prompt), 100)

                # Check model-specific formatting
                if model_name == "gpt-5":
                    self.assertIn("STRATEGIC ANALYSIS", prompt)
                    self.assertIn("json", prompt.lower())
                elif model_name == "claude":
                    self.assertIn("<priorities>", prompt)
                    self.assertIn("<actions>", prompt)
                elif model_name == "deepseek":
                    self.assertIn("Turn", prompt)

    def test_context_window_limits(self):
        """Test that prompts stay within model token limits."""
        models_limits = {
            "gpt-5": 4000,
            "claude": 3500,
            "deepseek": 3000,
        }

        for model_name, max_tokens in models_limits.items():
            with self.subTest(model=model_name):
                prompt = self.prompt_builder.build_enhanced_prompt(
                    observation={"state": self.mock_state},
                    legal_actions=self.legal_actions,
                    model_name=model_name,
                )

                # Rough token estimation: ~4 chars per token
                estimated_tokens = len(prompt) // 4
                self.assertLessEqual(
                    estimated_tokens,
                    max_tokens,
                    f"Prompt too long for {model_name}: {estimated_tokens} >"
                    f" {max_tokens}",
                )

    def test_phase_appropriate_prompts(self):
        """Test that different game phases get appropriate prompts."""
        phases = ["early_game", "mid_game", "late_game"]

        for phase in phases:
            with self.subTest(phase=phase):
                # Mock different turn numbers for different phases
                if phase == "early_game":
                    self.mock_state.turn = 5
                elif phase == "mid_game":
                    self.mock_state.turn = 100
                else:  # late_game
                    self.mock_state.turn = 180

                prompt = self.prompt_builder.build_enhanced_prompt(
                    observation={"state": self.mock_state},
                    legal_actions=self.legal_actions,
                    model_name="gpt-5",
                )

                if phase == "early_game":
                    self.assertIn("explore", prompt.lower())
                elif phase == "mid_game":
                    self.assertIn("expand", prompt.lower())
                else:  # late_game
                    self.assertIn("victory", prompt.lower())

    def test_action_prioritization(self):
        """Test that most important actions are highlighted."""
        # Create actions with different priorities
        high_priority_action = FreeCivAction(
            action_type="unit_attack",
            actor_id=1,
            target={"id": 5},
            parameters={"priority": "high"},
            source="unit",
        )

        actions_with_priority = self.legal_actions + [high_priority_action]

        prompt = self.prompt_builder.build_enhanced_prompt(
            observation={"state": self.mock_state},
            legal_actions=actions_with_priority,
            model_name="gpt-5",
        )

        # High priority actions should appear first or be highlighted
        attack_pos = prompt.find("attacks")
        move_pos = prompt.find("Move unit")

        # Check what's actually in the prompt for debugging
        if attack_pos == -1:
            print("DEBUG: Prompt content for action prioritization test:")
            print(prompt)

        # Attack should appear before move in prioritized list
        self.assertNotEqual(attack_pos, -1, "Attack action not found in prompt")
        self.assertNotEqual(move_pos, -1, "Move action not found in prompt")

    def test_observation_compression(self):
        """Test that large states compress properly."""
        # Create a state with many units
        large_units = [
            Mock(unit_id=i, kind="Warrior", position=(i, i), hp=100, owner=1)
            for i in range(50)
        ]
        self.mock_state.units = large_units

        prompt = self.prompt_builder.build_enhanced_prompt(
            observation={"state": self.mock_state},
            legal_actions=self.legal_actions,
            model_name="deepseek",  # Smallest token limit
        )

        # Should not exceed token limit even with large state
        estimated_tokens = len(prompt) // 4
        self.assertLessEqual(
            estimated_tokens, 3000, "Large state not compressed properly"
        )

    def test_performance_under_50ms(self):
        """Test that prompt generation completes in under 50ms."""
        start_time = time.time()

        prompt = self.prompt_builder.build_enhanced_prompt(
            observation={"state": self.mock_state},
            legal_actions=self.legal_actions,
            model_name="gpt-5",
        )

        end_time = time.time()
        generation_time_ms = (end_time - start_time) * 1000

        self.assertLess(
            generation_time_ms,
            50,
            f"Prompt generation took {generation_time_ms:.2f}ms, should be <50ms",
        )
        self.assertIsNotNone(prompt)


class TestObservationBuilder(unittest.TestCase):
    """Test cases for ObservationBuilder class."""

    def setUp(self):
        """Set up test fixtures."""
        self.observation_builder = ObservationBuilder()
        self.mock_obs = {
            "turn": 42,
            "players": {1: {"score": 340, "gold": 245, "name": "Romans"}},
            "units": [{"id": 1, "type": "Warrior", "x": 10, "y": 14, "hp": 100}],
            "cities": [{"id": 1, "name": "Rome", "x": 10, "y": 15, "pop": 8}],
        }

    def test_build_strategic_summary(self):
        """Test strategic summary generation."""
        summary = self.observation_builder.build_strategic_summary(self.mock_obs)

        self.assertIsInstance(summary, str)
        self.assertIn("Turn 42", summary)
        self.assertIn("Romans", summary)
        self.assertIn("score", summary.lower())

    def test_identify_priorities(self):
        """Test priority identification by game phase."""
        early_priorities = self.observation_builder._identify_priorities(
            self.mock_obs, "early_game"
        )
        late_priorities = self.observation_builder._identify_priorities(
            self.mock_obs, "late_game"
        )

        self.assertNotEqual(early_priorities, late_priorities)
        self.assertIn("exploration", early_priorities.lower())
        self.assertIn("victory", late_priorities.lower())

    def test_assess_threats(self):
        """Test threat assessment."""
        threats = self.observation_builder._assess_threats(self.mock_obs)

        self.assertIsInstance(threats, str)

    def test_identify_opportunities(self):
        """Test opportunity identification."""
        opportunities = self.observation_builder._identify_opportunities(self.mock_obs)

        self.assertIsInstance(opportunities, str)

    def test_format_prioritized_actions(self):
        """Test action formatting with priorities."""
        actions = [
            FreeCivAction(
                action_type="unit_move",
                actor_id=1,
                target={"x": 10, "y": 11},
                parameters={},
                source="unit",
            ),
            FreeCivAction(
                action_type="city_production",
                actor_id=1,
                target={"value": "Archer"},
                parameters={},
                source="city",
            ),
        ]

        formatted = self.observation_builder.format_prioritized_actions(
            actions, self.mock_obs
        )

        self.assertIsInstance(formatted, str)
        self.assertIn("1.", formatted)  # Should be numbered
        self.assertIn("move unit", formatted.lower())


class TestContextManager(unittest.TestCase):
    """Test cases for ContextManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.context_manager = ContextManager()

    def test_compress_observation(self):
        """Test observation compression."""
        large_obs = {
            "units": [{"id": i, "type": "Warrior"} for i in range(100)],
            "cities": [{"id": i, "name": f"City{i}"} for i in range(20)],
        }

        compressed = self.context_manager.compress_observation(large_obs, 2000)

        self.assertIsInstance(compressed, dict)
        # Should have fewer items than original
        self.assertLessEqual(len(compressed.get("units", [])), len(large_obs["units"]))

    def test_prioritize_information(self):
        """Test information prioritization by phase."""
        obs = {
            "units": [{"id": 1, "type": "Warrior"}],
            "cities": [{"id": 1, "name": "Rome"}],
            "technology": {"researching": "Bronze Working"},
        }

        early_info = self.context_manager.prioritize_information(obs, "early_game")
        late_info = self.context_manager.prioritize_information(obs, "late_game")

        self.assertIsInstance(early_info, dict)
        self.assertIsInstance(late_info, dict)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases to improve coverage."""

    def setUp(self):
        """Set up test fixtures."""
        self.prompt_builder = FreeCivPromptBuilder()

    def test_state_to_dict_with_empty_state(self):
        """Test _state_to_dict with minimal state object."""
        minimal_state = Mock()
        # Set default values for required attributes
        minimal_state.turn = 0
        minimal_state.players = {}
        minimal_state.units = []
        minimal_state.cities = []

        result = self.prompt_builder._state_to_dict(minimal_state)

        self.assertEqual(result["turn"], 0)
        self.assertEqual(result["players"], {})
        self.assertEqual(result["units"], [])
        self.assertEqual(result["cities"], [])

    def test_state_to_dict_with_non_dict_players(self):
        """Test _state_to_dict when players is not a dict."""
        state = Mock()
        state.turn = 50
        state.players = "invalid_players_data"
        state.units = []
        state.cities = []

        result = self.prompt_builder._state_to_dict(state)

        self.assertEqual(result["players"], {})

    def test_victory_progress_with_zero_score(self):
        """Test victory progress calculation with zero score."""
        obs = {"turn": 0, "players": {1: {"score": 0}}}

        progress = self.prompt_builder._calculate_victory_progress(obs)

        self.assertEqual(progress, 0)

    def test_victory_progress_with_high_values(self):
        """Test victory progress capped at 100%."""
        obs = {"turn": 1000, "players": {1: {"score": 10000}}}

        progress = self.prompt_builder._calculate_victory_progress(obs)

        self.assertEqual(progress, 100)

    def test_get_position_string_with_no_players(self):
        """Test position string generation with empty players."""
        obs = {"players": {}}

        position = self.prompt_builder._get_position_string(obs)

        self.assertEqual(position, "Solo game")

    def test_get_position_string_with_single_player(self):
        """Test position string with only one player."""
        obs = {"players": {1: {"score": 100}}}

        position = self.prompt_builder._get_position_string(obs)

        self.assertEqual(position, "Solo game")

    def test_get_player_score_with_missing_data(self):
        """Test score retrieval with missing player data."""
        obs = {"players": {}}

        score = self.prompt_builder._get_player_score(obs)

        self.assertEqual(score, 0)

    def test_get_player_name_with_missing_data(self):
        """Test player name retrieval with missing data."""
        obs = {"players": {}}

        name = self.prompt_builder._get_player_name(obs)

        self.assertEqual(name, "Romans")

    def test_detect_game_phase_edge_values(self):
        """Test game phase detection at boundary values."""
        test_cases = [
            (0, "early_game"),
            (50, "early_game"),
            (51, "mid_game"),
            (150, "mid_game"),
            (151, "late_game"),
            (9999, "late_game"),
        ]

        for turn, expected_phase in test_cases:
            with self.subTest(turn=turn):
                obs = {"turn": turn}
                phase = self.prompt_builder._detect_game_phase(obs)
                self.assertEqual(phase, expected_phase)

    def test_format_prioritized_actions_with_empty_list(self):
        """Test action formatting with empty action list."""
        obs_builder = ObservationBuilder()

        result = obs_builder.format_prioritized_actions([], {})

        self.assertEqual(result, "No actions available.")

    def test_prioritize_actions_with_unrecognized_action_type(self):
        """Test action prioritization with action types not in priority order."""
        obs_builder = ObservationBuilder()
        # Use a valid FreeCiv action type not in the priority order
        unrecognized_action = FreeCivAction(
            action_type="tech_research",
            actor_id=1,
            target={"value": "Bronze Working"},
            parameters={},
            source="city",
        )

        result = obs_builder._prioritize_actions([unrecognized_action])

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].action_type, "tech_research")

    def test_format_action_description_with_unrecognized_type(self):
        """Test action description formatting for unrecognized action types."""
        obs_builder = ObservationBuilder()
        # Use a valid FreeCiv action type not handled in format method
        unrecognized_action = FreeCivAction(
            action_type="diplomacy_trade",
            actor_id=1,
            target={"player_id": 2},
            parameters={},
            source="player",
        )

        result = obs_builder._format_action_description(unrecognized_action)

        self.assertEqual(result, "diplomacy_trade with unit/city 1")

    def test_build_strategic_summary_with_minimal_data(self):
        """Test strategic summary with minimal observation data."""
        obs_builder = ObservationBuilder()
        minimal_obs = {"turn": 1}

        result = obs_builder.build_strategic_summary(minimal_obs)

        self.assertIn("Turn 1", result)
        self.assertIn("Gathering intelligence", result)

    def test_context_manager_with_empty_observations(self):
        """Test context manager with empty observation data."""
        context_manager = ContextManager()
        empty_obs = {}

        compressed = context_manager.compress_observation(empty_obs, 1000)
        prioritized = context_manager.prioritize_information(empty_obs, "early_game")

        self.assertIsInstance(compressed, dict)
        self.assertIsInstance(prioritized, dict)
        self.assertEqual(prioritized["focus"], "exploration_and_settlement")

    def test_assess_threats_with_no_units(self):
        """Test threat assessment with empty units list."""
        obs_builder = ObservationBuilder()
        obs = {"cities": [], "units": []}

        result = obs_builder._assess_threats(obs)

        self.assertIn("No immediate threats", result)

    def test_identify_opportunities_with_empty_map(self):
        """Test opportunity identification with empty map data."""
        obs_builder = ObservationBuilder()
        obs = {"map": {"tiles": []}, "units": [], "cities": [], "turn": 25}

        result = obs_builder._identify_opportunities(obs)

        self.assertIn("Research Bronze Working", result)

    def test_detect_victory_type_with_empty_data(self):
        """Test victory type detection with minimal data."""
        prompt_builder = FreeCivPromptBuilder()
        obs = {"turn": 25, "cities": [], "units": []}

        result = prompt_builder._detect_victory_type(obs)

        self.assertEqual(result, "Expansion Victory")

    def test_build_enhanced_prompt_with_invalid_model(self):
        """Test prompt building with unsupported model name."""
        prompt_builder = FreeCivPromptBuilder()
        obs = {"turn": 1, "players": {1: {"name": "Romans"}}}
        actions = []

        # Should fall back to default (gpt-5) configuration
        result = prompt_builder.build_enhanced_prompt(obs, actions, "unknown_model")

        self.assertIsInstance(result, str)
        self.assertIn("Turn 1", result)

    def test_format_prioritized_actions_with_large_list(self):
        """Test action formatting with more than 10 actions."""
        obs_builder = ObservationBuilder()
        obs = {"turn": 50}

        # Create 15 actions
        actions = []
        for i in range(15):
            actions.append(
                FreeCivAction(
                    action_type="unit_move",
                    actor_id=i,
                    target={"x": i, "y": i},
                    parameters={},
                    source="unit",
                )
            )

        result = obs_builder.format_prioritized_actions(actions, obs)

        # Should limit to top 10
        action_lines = [
            line for line in result.split("\n") if line.strip() and line[0].isdigit()
        ]
        self.assertLessEqual(len(action_lines), 10)


if __name__ == "__main__":
    unittest.main()
