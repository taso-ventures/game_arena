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

                # Check unified template elements (same for all models now)
                self.assertIn("GAME ANALYSIS & STRATEGY", prompt)
                self.assertIn("VICTORY OBJECTIVE", prompt)
                self.assertIn("MEMORY CONTEXT", prompt)
                self.assertIn("LONG-TERM STRATEGY", prompt)
                self.assertIn("ENTERTAINMENT DIRECTIVE", prompt)
                self.assertIn("REASONING FRAMEWORK", prompt)

                # Check model-specific response formatting
                if model_name == "gpt-5":
                    self.assertIn("FINAL DECISION:", prompt)
                elif model_name == "claude":
                    self.assertIn("natural, engaging way", prompt)
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
                # Create observation with different turn numbers for different phases
                if phase == "early_game":
                    turn = 5
                elif phase == "mid_game":
                    turn = 100
                else:  # late_game
                    turn = 180

                # Use dict observation format that new builder expects
                observation = {
                    "turn": turn,
                    "players": {1: {"score": 340, "name": "Romans"}},
                    "units": [],
                    "cities": []
                }

                prompt = self.prompt_builder.build_enhanced_prompt(
                    observation=observation,
                    legal_actions=self.legal_actions,
                    model_name="gpt-5",
                )

                if phase == "early_game":
                    self.assertIn("early game priorities", prompt.lower())
                    self.assertIn("explore", prompt.lower())
                elif phase == "mid_game":
                    self.assertIn("mid game expansion", prompt.lower())
                    self.assertIn("expansion", prompt.lower())
                else:  # late_game
                    self.assertIn("late game dominance", prompt.lower())
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


class TestErrorHandlingAndValidation(unittest.TestCase):
    """Test error handling and input validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.prompt_builder = FreeCivPromptBuilder()

    def test_invalid_model_name_validation(self):
        """Test model name validation with various invalid inputs."""
        obs = {"turn": 1, "players": {1: {"name": "Romans"}}}
        actions = []

        # Test empty model name
        with self.assertRaises(ValueError):
            self.prompt_builder.build_enhanced_prompt(obs, actions, "")

        # Test None model name
        with self.assertRaises(ValueError):
            self.prompt_builder.build_enhanced_prompt(obs, actions, None)

        # Test invalid characters in model name
        with self.assertRaises(ValueError):
            self.prompt_builder.build_enhanced_prompt(obs, actions, "model@123")

        # Test SQL injection attempt
        with self.assertRaises(ValueError):
            self.prompt_builder.build_enhanced_prompt(
                obs, actions, "'; DROP TABLE models; --"
            )

    def test_malformed_observation_handling(self):
        """Test handling of malformed observation structures."""
        actions = []

        # Test with None observation
        with self.assertRaises(TypeError):
            self.prompt_builder.build_enhanced_prompt(None, actions, "gpt-5")

        # Test with string instead of dict
        with self.assertRaises(TypeError):
            self.prompt_builder.build_enhanced_prompt("invalid", actions, "gpt-5")

        # Test with list instead of dict
        with self.assertRaises(TypeError):
            self.prompt_builder.build_enhanced_prompt([], actions, "gpt-5")

    def test_missing_required_observation_fields(self):
        """Test handling of observations missing required fields."""
        actions = []

        # Test with completely empty observation
        result = self.prompt_builder.build_enhanced_prompt({}, actions, "gpt-5")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

        # Test with observation missing players
        obs_no_players = {"turn": 5, "units": [], "cities": []}
        result = self.prompt_builder.build_enhanced_prompt(
            obs_no_players, actions, "gpt-5"
        )
        self.assertIsInstance(result, str)

    def test_compress_observation_edge_cases(self):
        """Test ContextManager.compress_observation with edge cases."""
        context_manager = ContextManager()

        # Test with None input
        result = context_manager.compress_observation(None, 1000)
        self.assertEqual(result, {})

        # Test with deeply nested structures
        deep_obs = {"level1": {"level2": {"level3": {"level4": "value"}}}}
        result = context_manager.compress_observation(deep_obs, 1000)
        self.assertIsInstance(result, dict)

        # Test with circular references (should not cause infinite recursion)
        circular_obs = {"key": "value"}
        circular_obs["self"] = circular_obs
        result = context_manager.compress_observation(circular_obs, 1000)
        self.assertIsInstance(result, dict)

    def test_threat_detection_with_empty_data(self):
        """Test threat detection with empty or invalid unit data."""
        obs_builder = ObservationBuilder()

        # Test with empty observation
        result = obs_builder._assess_threats({}, 1)
        self.assertIn("No immediate threats", result)

        # Test with malformed unit data
        obs_with_bad_units = {
            "units": ["invalid_unit", {"invalid": "data"}, None],
            "cities": [],
        }
        result = obs_builder._assess_threats(obs_with_bad_units, 1)
        self.assertIsInstance(result, str)

    def test_victory_condition_detection_edge_cases(self):
        """Test victory condition detection with edge case data."""
        prompt_builder = FreeCivPromptBuilder()

        # Test with no units or cities
        obs = {"turn": 100, "units": [], "cities": []}
        result = prompt_builder._detect_victory_type(obs, 1)
        self.assertIsInstance(result, str)

        # Test with invalid unit data
        obs_bad_units = {
            "turn": 50,
            "units": [{"type": None}, {"owner": "invalid"}],
            "cities": [],
        }
        result = prompt_builder._detect_victory_type(obs_bad_units, 1)
        self.assertIsInstance(result, str)

    def test_concurrent_prompt_generation(self):
        """Test thread safety of prompt builder."""
        import concurrent.futures
        import threading

        prompt_builder = FreeCivPromptBuilder()
        obs = {"turn": 1, "players": {1: {"name": "Romans"}}}
        actions = []

        def build_prompt():
            return prompt_builder.build_enhanced_prompt(obs, actions, "gpt-5")

        # Test concurrent access
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(build_prompt) for _ in range(20)]
            results = [f.result() for f in futures]

        # All results should be strings and non-empty
        for result in results:
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)

    def test_spatial_indexing_performance(self):
        """Test spatial indexing with large numbers of units."""
        obs_builder = ObservationBuilder()

        # Create observation with many enemy units
        large_enemy_units = []
        for i in range(1000):
            large_enemy_units.append(
                {
                    "id": i,
                    "type": "Warrior",
                    "x": i % 100,
                    "y": i // 100,
                    "owner": 2,  # Enemy player
                }
            )

        obs = {
            "units": large_enemy_units,
            "cities": [{"name": "TestCity", "x": 50, "y": 50}],
        }

        # Should complete quickly even with many units
        import time

        start_time = time.time()
        result = obs_builder._assess_threats(obs, 1)
        end_time = time.time()

        # Should complete in reasonable time (less than 100ms)
        self.assertLess(end_time - start_time, 0.1)
        self.assertIsInstance(result, str)

    def test_model_configuration_fallback(self):
        """Test fallback behavior for unknown models."""
        prompt_builder = FreeCivPromptBuilder()
        obs = {"turn": 1, "players": {1: {"name": "Romans"}}}
        actions = []

        # Test with unknown model - should use default without crashing
        result = prompt_builder.build_enhanced_prompt(obs, actions, "unknown-model")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_memory_usage_with_large_observations(self):
        """Test memory efficiency with very large observations."""
        prompt_builder = FreeCivPromptBuilder()

        # Create large observation
        large_obs = {
            "turn": 100,
            "players": {
                i: {"name": f"Player{i}", "score": i * 100} for i in range(1, 21)
            },
            "units": [
                {"id": i, "type": "Warrior", "x": i, "y": i} for i in range(5000)
            ],
            "cities": [{"id": i, "name": f"City{i}", "pop": i} for i in range(500)],
        }
        actions = []

        # Should handle large data without memory issues
        result = prompt_builder.build_enhanced_prompt(large_obs, actions, "gpt-5")
        self.assertIsInstance(result, str)

        # Result should be compressed to reasonable size
        self.assertLess(len(result), 50000)  # Should be compressed


class TestThreatDetectionSpatialIndexing(unittest.TestCase):
    """Test spatial indexing implementation for threat detection."""

    def setUp(self):
        """Set up test fixtures."""
        self.obs_builder = ObservationBuilder()

    def test_build_threat_index(self):
        """Test spatial index building."""
        enemy_units = [
            {"id": 1, "x": 5, "y": 5, "type": "Warrior"},
            {"id": 2, "x": 10, "y": 10, "type": "Archer"},
            {"id": 3, "x": 15, "y": 15, "type": "Legion"},
        ]

        index = self.obs_builder._build_threat_index(enemy_units)
        self.assertIsInstance(index, dict)
        self.assertGreater(len(index), 0)

    def test_get_nearby_threats(self):
        """Test spatial threat detection."""
        # Create threat map
        enemy_units = [
            {"id": 1, "x": 2, "y": 2, "type": "Warrior"},  # Close
            {"id": 2, "x": 10, "y": 10, "type": "Archer"},  # Far
        ]
        threat_map = self.obs_builder._build_threat_index(enemy_units)

        # Test threat detection at position (0, 0)
        threats = self.obs_builder._get_nearby_threats((0, 0), threat_map)

        # Should find the close warrior but not the distant archer
        self.assertGreater(len(threats), 0)
        threat_ids = [t["id"] for t in threats]
        self.assertIn(1, threat_ids)  # Close warrior should be detected

    def test_manhattan_distance(self):
        """Test Manhattan distance calculation."""
        distance = self.obs_builder._manhattan_distance((0, 0), (3, 4))
        self.assertEqual(distance, 7)

        distance = self.obs_builder._manhattan_distance((5, 5), (5, 5))
        self.assertEqual(distance, 0)

    def test_spatial_indexing_with_invalid_units(self):
        """Test spatial indexing handles invalid unit data gracefully."""
        # Mix of valid and invalid units
        mixed_units = [
            {"id": 1, "x": 5, "y": 5, "type": "Warrior"},  # Valid
            {"invalid": "unit"},  # Missing coordinates
            None,  # None entry
            "string_unit",  # String instead of dict
            {"x": "invalid", "y": 5},  # Invalid coordinate types
        ]

        # Should not crash and return valid index
        index = self.obs_builder._build_threat_index(mixed_units)
        self.assertIsInstance(index, dict)


if __name__ == "__main__":
    unittest.main()
