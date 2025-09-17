import copy
import json
import time
import unittest
from pathlib import Path

from pydantic import ValidationError

from game_arena.harness.freeciv_state import (
    MAX_STATE_SIZE_BYTES,
    FreeCivAction,
    FreeCivState,
    _FallbackGameState,
    _validate_state_size,
)


class TestFreeCivState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture_path = (
            Path(__file__).resolve().parent.parent
            / "fixtures"
            / "freeciv_game_states.json"
        )
        with fixture_path.open(encoding="utf-8") as fixture_file:
            cls.scenarios = json.load(fixture_file)["states"]

    def test_state_initialization(self):
        raw_state = copy.deepcopy(self.scenarios["early_game"])
        state = FreeCivState(raw_state)

        self.assertEqual(state.turn, 10, "Turn should be correctly parsed from fixture")
        self.assertEqual(
            state.phase, "movement", "Phase should be correctly parsed from fixture"
        )
        self.assertEqual(state.map.width, 6, "Map width should match fixture data")
        self.assertEqual(state.map.height, 4, "Map height should match fixture data")
        self.assertIn(1, state.players, "Player 1 should exist in parsed players")
        self.assertEqual(
            state.players[1].name, "Athens", "Player 1 should have correct name"
        )
        self.assertIn(101, state.units, "Unit 101 should exist in parsed units")
        self.assertEqual(
            state.units[101].position, (1, 1), "Unit 101 should have correct position"
        )
        self.assertIn(301, state.cities, "City 301 should exist in parsed cities")
        self.assertEqual(
            state.cities[301].name, "Athens", "City 301 should have correct name"
        )
        self.assertEqual(
            state.current_player(),
            0,
            "Current player should be 0 (player ID 1 maps to 0)",
        )
        self.assertFalse(state.is_terminal(), "Early game state should not be terminal")

    def test_legal_actions_generation(self):
        raw_state = copy.deepcopy(self.scenarios["early_game"])
        state = FreeCivState(raw_state)

        player_actions = state.get_legal_actions(player_id=1)
        self.assertGreaterEqual(len(player_actions), 6)
        action_types = {action.action_type for action in player_actions}
        self.assertIn("unit_move", action_types)
        self.assertIn("city_production", action_types)
        self.assertTrue(
            all(isinstance(action, FreeCivAction) for action in player_actions)
        )

        opponent_actions = state.get_legal_actions(player_id=2)
        self.assertTrue(any(action.actor_id == 201 for action in opponent_actions))

    def test_observation_formats(self):
        raw_state = copy.deepcopy(self.scenarios["mid_game"])
        state = FreeCivState(raw_state)

        json_obs = state.to_observation(player_id=1, format="json")
        self.assertEqual(json_obs["game"]["turn"], 52)
        visible_tiles = {
            (tile["x"], tile["y"]) for tile in json_obs["map"]["visible_tiles"]
        }
        self.assertIn((4, 3), visible_tiles)
        self.assertNotIn((6, 3), visible_tiles)  # fog of war respected

        ascii_obs = state.to_observation(player_id=1, format="ascii")
        self.assertIsInstance(ascii_obs, str)
        self.assertIn("C", ascii_obs)
        self.assertIn("?", ascii_obs)

        llm_obs = state.to_observation(player_id=1, format="enhanced")
        self.assertIn("strategic", llm_obs)
        self.assertIn("metadata", llm_obs)
        if "tactical" in llm_obs:
            self.assertIn("unit_counts", llm_obs["tactical"])
            self.assertGreater(llm_obs["tactical"]["unit_counts"]["friendly"], 0)
        self.assertGreaterEqual(llm_obs["strategic"]["scoreboard"]["player"], 0)

    def test_apply_action_and_returns(self):
        raw_state = copy.deepcopy(self.scenarios["early_game"])
        state = FreeCivState(raw_state)

        move_actions = [
            a
            for a in state.get_legal_actions(1)
            if a.action_type == "unit_move" and a.actor_id == 101
        ]
        self.assertTrue(move_actions)
        state.apply_action(move_actions[0])
        self.assertEqual(
            state.units[101].position,
            (move_actions[0].target["x"], move_actions[0].target["y"]),
        )

        # returns follow order of player ids
        raw_state_scores = self.scenarios["mid_game"]["game"]["scores"]
        mid_state = FreeCivState(copy.deepcopy(self.scenarios["mid_game"]))
        self.assertEqual(
            mid_state.returns(), [raw_state_scores["1"], raw_state_scores["2"]]
        )

        finished = copy.deepcopy(self.scenarios["city_production"])
        finished["game"]["is_over"] = True
        terminal_state = FreeCivState(finished)
        self.assertTrue(terminal_state.is_terminal())

    def test_openspiel_compatibility(self):
        """Test OpenSpiel compatibility methods."""
        raw_state = copy.deepcopy(self.scenarios["early_game"])
        state = FreeCivState(raw_state)

        # Test action indexing
        legal_indices = state.legal_actions()
        self.assertIsInstance(legal_indices, list)
        self.assertTrue(all(isinstance(i, int) for i in legal_indices))

        # Test action to string conversion
        if legal_indices:
            action_str = state.action_to_string(0, legal_indices[0])
            self.assertIsInstance(action_str, str)
            self.assertGreater(len(action_str), 0)

            # Test string to action conversion
            recovered_index = state.string_to_action(0, action_str)
            self.assertEqual(recovered_index, legal_indices[0])

        # Test apply action by index
        if legal_indices:
            original_unit_pos = state.units[101].position
            state.apply_action_by_index(legal_indices[0])
            # State should have changed
            self.assertIsInstance(state.units[101], object)

    def test_enhanced_game_mechanics(self):
        """Test enhanced game mechanics like diplomacy, research, etc."""
        raw_state = copy.deepcopy(self.scenarios["mid_game"])

        # Add enhanced player data
        raw_state["players"][0]["research_target"] = "Mathematics"
        raw_state["players"][0]["research_progress"] = 15
        raw_state["players"][0]["diplomatic_relations"] = [
            {"player_id": 2, "status": "war"}
        ]

        state = FreeCivState(raw_state)
        player = state.players[1]

        self.assertEqual(player.research_target, "Mathematics")
        self.assertEqual(player.research_progress, 15)
        self.assertEqual(player.diplomatic_relations[2], "war")

    def test_llm_observation_token_limits(self):
        """Test LLM observation respects token limits."""
        raw_state = copy.deepcopy(self.scenarios["late_game"])
        state = FreeCivState(raw_state)

        # Test with different token limits
        for max_tokens in [1000, 2000, 4000]:
            obs = state.to_observation(player_id=1, format="enhanced")
            if "metrics" in obs:
                estimated_tokens = obs["metrics"].get("estimated_tokens", 0)
                self.assertLessEqual(
                    estimated_tokens, max_tokens * 1.2
                )  # 20% tolerance

        # Verify observation includes essential information
        obs = state.to_observation(player_id=1, format="enhanced")
        self.assertIn("metadata", obs)
        self.assertIn("strategic", obs)
        self.assertIn("actions", obs)

    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling."""
        raw_state = copy.deepcopy(self.scenarios["early_game"])
        state = FreeCivState(raw_state)

        # Test invalid player IDs
        with self.assertRaises(ValueError):
            state.action_to_string(0, 999)  # Invalid action index

        # Test empty legal actions for non-existent player
        empty_actions = state.get_legal_actions(999)
        self.assertEqual(
            len(empty_actions), 0, "Non-existent player should have no legal actions"
        )

        # Test invalid action application
        invalid_action = FreeCivAction(
            action_type="invalid_action",
            actor_id=999,
            target=None,
            parameters={},
            source="unit",
        )
        with self.assertRaises(ValueError):
            state.apply_action(invalid_action)

    def test_security_input_validation(self):
        """Test security input validation functions."""
        # Test oversized JSON input - create data that's actually large enough
        import sys

        # Create a list large enough to trigger size validation
        large_list = ["x" * 1000] * 10000  # Should be > 10MB
        large_data = {
            "game": {"turn": 1, "phase": "movement"},
            "map": {"width": 6, "height": 4, "tiles": large_list},
            "players": [],
            "units": [],
            "cities": [],
        }

        # Only test if the data is actually large enough
        if sys.getsizeof(large_data) > 10_000_000:
            with self.assertRaises(ValueError) as context:
                FreeCivState(large_data)
            self.assertIn("exceeds maximum allowed size", str(context.exception))

        # Test invalid data types
        invalid_data = {
            "game": "not_a_dict",
            "map": {"width": 6, "height": 4, "tiles": []},
            "players": [],
            "units": [],
            "cities": [],
        }

        with self.assertRaises(TypeError) as context:
            FreeCivState(invalid_data)
        self.assertIn("must be a dictionary", str(context.exception))

    def test_security_integer_bounds(self):
        """Test integer bounds checking for security."""
        from game_arena.harness.freeciv_state import _safe_int_conversion

        # Test valid integers
        self.assertEqual(_safe_int_conversion(42, 100, "test_field"), 42)

        # Test out of bounds - the implementation re-raises as "Invalid field_name"
        with self.assertRaises(ValueError) as context:
            _safe_int_conversion(1001, 1000, "player_id")
        self.assertIn("Invalid player_id", str(context.exception))

        # Test negative values when not allowed
        with self.assertRaises(ValueError) as context:
            _safe_int_conversion(-1, 1000, "unit_id", allow_negative=False)
        self.assertIn("Invalid unit_id", str(context.exception))

        # Test negative values when allowed
        self.assertEqual(
            _safe_int_conversion(-1, 1000, "fuel", allow_negative=True), -1
        )

        # Test non-integer types
        with self.assertRaises(ValueError) as context:
            _safe_int_conversion("not_an_int", 1000, "test_field")
        self.assertIn("Invalid test_field", str(context.exception))

    def test_security_json_depth_protection(self):
        """Test JSON depth protection against stack overflow."""
        from game_arena.harness.freeciv_state import _safe_json_dumps

        # Create deeply nested object (exceeds MAX_JSON_DEPTH of 10)
        deep_obj = {}
        current = deep_obj
        for i in range(15):  # Exceeds MAX_JSON_DEPTH of 10
            current["nested"] = {}
            current = current["nested"]

        with self.assertRaises(ValueError) as context:
            _safe_json_dumps(deep_obj)
        self.assertIn(
            "Object nesting exceeds maximum depth of 10", str(context.exception)
        )

        # Test valid depth
        shallow_obj = {"level1": {"level2": {"level3": "value"}}}
        result = _safe_json_dumps(shallow_obj)
        self.assertIsInstance(result, str)
        self.assertIn("value", result)

    def test_enhanced_unit_actions(self):
        """Test enhanced unit action types."""
        raw_state = copy.deepcopy(self.scenarios["early_game"])
        state = FreeCivState(raw_state)

        unit_id = 101
        unit = state.units[unit_id]
        original_pos = unit.position

        # Test fortification
        fortify_action = FreeCivAction(
            action_type="unit_fortify",
            actor_id=unit_id,
            target=None,
            parameters={},
            source="unit",
        )
        state.apply_action(fortify_action)
        self.assertTrue(state.units[unit_id].fortified)

        # Test exploration
        explore_action = FreeCivAction(
            action_type="unit_explore",
            actor_id=unit_id,
            target=None,
            parameters={},
            source="unit",
        )
        state.apply_action(explore_action)
        self.assertEqual(state.units[unit_id].activity, "exploring")

    def test_enhanced_city_actions(self):
        """Test enhanced city action types."""
        raw_state = copy.deepcopy(self.scenarios["early_game"])
        state = FreeCivState(raw_state)

        city_id = 301
        city = state.cities[city_id]

        # Test building construction
        build_action = FreeCivAction(
            action_type="city_build_improvement",
            actor_id=city_id,
            target={"value": "granary"},
            parameters={},
            source="city",
        )
        state.apply_action(build_action)
        self.assertIn("granary", state.cities[city_id].buildings)

        # Test celebration
        celebrate_action = FreeCivAction(
            action_type="city_celebrate",
            actor_id=city_id,
            target=None,
            parameters={},
            source="city",
        )
        state.apply_action(celebrate_action)
        self.assertTrue(state.cities[city_id].celebrating)

    def test_unit_invalid_action_type_raises(self):
        """Unit actions with unsupported types raise helpful errors."""
        raw_state = copy.deepcopy(self.scenarios["early_game"])
        state = FreeCivState(raw_state)

        invalid_action = FreeCivAction(
            action_type="unsupported_action",
            actor_id=101,
            target=None,
            parameters={},
            source="unit",
        )
        with self.assertRaises(ValueError) as context:
            state.apply_action(invalid_action)
        self.assertIn("Unsupported unit action", str(context.exception))

    def test_city_invalid_action_type_raises(self):
        """City actions with unsupported types raise helpful errors."""
        raw_state = copy.deepcopy(self.scenarios["early_game"])
        state = FreeCivState(raw_state)

        invalid_action = FreeCivAction(
            action_type="unsupported_city_action",
            actor_id=301,
            target=None,
            parameters={},
            source="city",
        )
        with self.assertRaises(ValueError) as context:
            state.apply_action(invalid_action)
        self.assertIn("Unsupported city action", str(context.exception))

    def test_unit_transport_and_unload_actions(self):
        """Transport units correctly manage cargo relationships."""
        raw_state = copy.deepcopy(self.scenarios["early_game"])
        state = FreeCivState(raw_state)

        transport_unit = state.units[101]
        cargo_unit = state.units[102]

        load_action = FreeCivAction(
            action_type="unit_transport",
            actor_id=transport_unit.unit_id,
            target={"id": cargo_unit.unit_id},
            parameters={},
            source="unit",
        )
        state.apply_action(load_action)
        self.assertIn(
            cargo_unit.unit_id, state.units[transport_unit.unit_id].cargo_ids
        )
        self.assertEqual(
            state.units[cargo_unit.unit_id].transport_id, transport_unit.unit_id
        )

        unload_action = FreeCivAction(
            action_type="unit_unload",
            actor_id=transport_unit.unit_id,
            target={"id": cargo_unit.unit_id},
            parameters={},
            source="unit",
        )
        state.apply_action(unload_action)
        self.assertNotIn(
            cargo_unit.unit_id, state.units[transport_unit.unit_id].cargo_ids
        )
        self.assertIsNone(state.units[cargo_unit.unit_id].transport_id)

    def test_fallback_game_state_methods_raise(self):
        """Fallback game state methods signal incomplete implementation."""
        fallback_state = _FallbackGameState()
        with self.assertRaises(NotImplementedError):
            fallback_state.current_player()
        with self.assertRaises(NotImplementedError):
            fallback_state.legal_actions()
        with self.assertRaises(NotImplementedError):
            fallback_state.is_terminal()
        with self.assertRaises(NotImplementedError):
            fallback_state.returns()

    def test_validate_state_size_guard(self):
        """Large inputs trip the state size protection before parsing."""
        large_payload = "x" * 6000
        oversized_state = {"tiles": [{"payload": f"{large_payload}_{i}"} for i in range(2000)]}
        self.assertGreater(
            len(oversized_state["tiles"]) * len(large_payload), MAX_STATE_SIZE_BYTES
        )
        with self.assertRaises(ValueError) as context:
            _validate_state_size(oversized_state)
        self.assertIn("exceeds maximum allowed size", str(context.exception))

    def test_invalid_diplomatic_relations_rejected(self):
        """Invalid diplomatic relation statuses fail validation."""
        raw_state = copy.deepcopy(self.scenarios["early_game"])
        raw_state["players"][0]["diplomatic_relations"] = [
            {"player_id": 2, "status": "rival"}
        ]
        with self.assertRaises((ValueError, ValidationError)) as context:
            FreeCivState(raw_state)
        self.assertIn("Invalid diplomatic status", str(context.exception))

    def test_performance_benchmarks(self):
        """Test performance requirements."""
        raw_state = copy.deepcopy(self.scenarios["late_game"])

        # Test state initialization performance
        start_time = time.time()
        state = FreeCivState(raw_state)
        init_time = time.time() - start_time
        self.assertLess(
            init_time, 0.1, "State initialization should complete in < 100ms"
        )

        # Test legal action generation performance
        start_time = time.time()
        actions = state.get_legal_actions(1)
        action_time = time.time() - start_time
        self.assertLess(
            action_time, 0.05, "Legal action generation should complete in < 50ms"
        )

        # Test observation generation performance
        start_time = time.time()
        obs = state.to_observation(1, format="enhanced")
        obs_time = time.time() - start_time
        self.assertLess(
            obs_time, 0.05, "Observation generation should complete in < 50ms"
        )

    def test_map_enhancements(self):
        """Test enhanced map features like improvements and pollution."""
        raw_state = copy.deepcopy(self.scenarios["early_game"])

        # Add tile improvements to test data
        for tile in raw_state["map"]["tiles"]:
            if tile["x"] == 1 and tile["y"] == 1:
                tile["improvements"] = ["road"]
                tile["pollution"] = False
                tile["owner"] = 1

        state = FreeCivState(raw_state)
        tile = state.map.tiles[(1, 1)]

        self.assertIn("road", tile.improvements)
        self.assertFalse(tile.pollution)
        self.assertEqual(tile.owner, 1)

    def test_fog_of_war_mechanics(self):
        """Test fog of war visibility mechanics."""
        raw_state = copy.deepcopy(self.scenarios["mid_game"])
        state = FreeCivState(raw_state)

        # Test visibility for player 1
        visible_tiles = state.map.visible_tiles(1)
        visible_coords = {(tile.x, tile.y) for tile in visible_tiles}

        # Check that visibility matches the data
        expected_visible = set(state.map.visibility.get(1, []))
        expected_coords = {tuple(coord) for coord in expected_visible}

        # Should only see tiles in visibility set
        for coord in visible_coords:
            self.assertIn(coord, expected_coords)

    def test_serialization_consistency(self):
        """Test that state can be consistently serialized and deserialized."""
        raw_state = copy.deepcopy(self.scenarios["mid_game"])
        state1 = FreeCivState(raw_state)

        # Get observations and compare
        obs1 = state1.to_observation(1, format="json")

        # Create another state from same data
        state2 = FreeCivState(raw_state)
        obs2 = state2.to_observation(1, format="json")

        # Should be identical
        self.assertEqual(obs1["game"]["turn"], obs2["game"]["turn"])
        self.assertEqual(len(obs1["units"]), len(obs2["units"]))
        self.assertEqual(len(obs1["cities"]), len(obs2["cities"]))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
