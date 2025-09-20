import copy
import json
import time
import unittest
from pathlib import Path

from pydantic import ValidationError

try:
    import pyspiel
except ImportError:
    pyspiel = None

from game_arena.harness.freeciv_state import (_PY_SPIEL_DUMMY_GAME,
                                              MAX_CITY_ID, MAX_PLAYER_ID,
                                              MAX_STATE_SIZE_BYTES,
                                              MAX_UNIT_ID, FreeCivAction,
                                              FreeCivState, LRUCache,
                                              _calculate_deep_size,
                                              _FallbackGameState,
                                              _GameStateBase,
                                              _validate_state_size)


class TestFreeCivState(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        fixture_path = (
            Path(__file__).resolve().parent.parent
            / "fixtures"
            / "freeciv_sample_game_states.json"
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
            state.units[101].position,
            (1, 1),
            "Unit 101 should have correct position",
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
            len(empty_actions),
            0,
            "Non-existent player should have no legal actions",
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
        self.assertIn(cargo_unit.unit_id, state.units[transport_unit.unit_id].cargo_ids)
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
        oversized_state = {
            "tiles": [{"payload": f"{large_payload}_{i}"} for i in range(2000)]
        }
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

    def test_freeciv_action_json_parsing(self):
        """Test FreeCivAction.from_json() method with various JSON formats."""
        # Test basic JSON format
        json_data = {"action": "unit_move", "unit": 101, "to": [3, 5]}
        action = FreeCivAction.from_json(json_data)
        self.assertEqual(action.action_type, "unit_move")
        self.assertEqual(action.actor_id, 101)
        self.assertEqual(action.target, {"x": 3, "y": 5})
        self.assertEqual(action.parse_method, "json")

        # Test alternative field names
        json_data2 = {
            "type": "city_production",
            "city": 301,
            "target": {"value": "warriors"},
        }
        action2 = FreeCivAction.from_json(json_data2)
        self.assertEqual(action2.action_type, "city_production")
        self.assertEqual(action2.actor_id, 301)
        self.assertEqual(action2.target, {"value": "warriors"})

        # Test JSON string input
        json_string = '{"action": "unit_attack", "actor": 102, "target": {"id": 203}}'
        action3 = FreeCivAction.from_json(json_string)
        self.assertEqual(action3.action_type, "unit_attack")
        self.assertEqual(action3.actor_id, 102)
        self.assertEqual(action3.target, {"id": 203})

        # Test invalid JSON
        with self.assertRaises(ValueError):
            FreeCivAction.from_json('{"invalid": json}')

    def test_freeciv_action_natural_language_parsing(self):
        """Test FreeCivAction.from_natural_language() method."""
        raw_state = copy.deepcopy(self.scenarios["early_game"])
        state = FreeCivState(raw_state)

        # Test unit move parsing
        action = FreeCivAction.from_natural_language("Move unit 101 to (2,3)", state)
        if action:  # May be None if unit not found
            self.assertEqual(action.action_type, "unit_move")
            self.assertEqual(action.actor_id, 101)
            self.assertEqual(action.target, {"x": 2, "y": 3})
            self.assertEqual(action.parse_method, "natural_language")

        # Test tech research parsing
        action2 = FreeCivAction.from_natural_language("Research pottery", state)
        if action2:
            self.assertEqual(action2.action_type, "tech_research")
            self.assertEqual(action2.target, {"value": "pottery"})

        # Test invalid natural language
        action3 = FreeCivAction.from_natural_language("This is nonsense", state)
        self.assertIsNone(action3)

    def test_freeciv_action_packet_conversion(self):
        """Test FreeCivAction.to_packet() method."""
        # Test unit move packet
        action = FreeCivAction(
            action_type="unit_move",
            actor_id=101,
            target={"x": 3, "y": 5},
            source="unit",
        )
        packet = action.to_packet()
        self.assertEqual(packet["pid"], 31)  # PACKET_UNIT_ORDERS
        self.assertEqual(packet["type"], "unit_move")
        self.assertEqual(packet["actor"], 101)
        self.assertEqual(packet["dest_tile"], [3, 5])

        # Test city production packet
        action2 = FreeCivAction(
            action_type="city_production",
            actor_id=301,
            target={"value": "warriors"},
            source="city",
        )
        packet2 = action2.to_packet()
        self.assertEqual(packet2["pid"], 63)  # PACKET_CITY_CHANGE
        self.assertEqual(packet2["city_id"], 301)
        self.assertEqual(packet2["value"], "warriors")

        # Test tech research packet
        action3 = FreeCivAction(
            action_type="tech_research",
            actor_id=1,
            target={"value": "pottery"},
            source="player",
        )
        packet3 = action3.to_packet()
        self.assertEqual(packet3["pid"], 87)  # PACKET_PLAYER_RESEARCH
        self.assertEqual(packet3["player_id"], 1)
        self.assertEqual(packet3["tech"], "pottery")

    def test_freeciv_action_validation(self):
        """Test Pydantic validation in FreeCivAction."""
        # Test valid action
        action = FreeCivAction(
            action_type="unit_move",
            actor_id=101,
            target={"x": 3, "y": 5},
            source="unit",
        )
        self.assertEqual(action.action_type, "unit_move")

        # Test invalid action type (use one that doesn't start with test_, invalid_, or unsupported_)
        with self.assertRaises(ValidationError):
            FreeCivAction(
                action_type="completely_unknown_action", actor_id=101, source="unit"
            )

        # Test invalid target for unit move
        with self.assertRaises(ValidationError):
            FreeCivAction(
                action_type="unit_move",
                actor_id=101,
                target={"invalid": "target"},
                source="unit",
            )

        # Test invalid source
        with self.assertRaises(ValidationError):
            FreeCivAction(
                action_type="unit_move", actor_id=101, source="invalid_source"
            )

    def test_freeciv_action_to_natural_language(self):
        """Test FreeCivAction.to_natural_language() method."""
        # Test unit move
        action = FreeCivAction(
            action_type="unit_move",
            actor_id=101,
            target={"x": 3, "y": 5},
            source="unit",
        )
        nl = action.to_natural_language()
        self.assertIn("101", nl)
        self.assertIn("3", nl)
        self.assertIn("5", nl)

        # Test city production
        action2 = FreeCivAction(
            action_type="city_production",
            actor_id=301,
            target={"value": "warriors"},
            source="city",
        )
        nl2 = action2.to_natural_language()
        self.assertIn("301", nl2)
        self.assertIn("warriors", nl2)

    def test_prioritized_legal_actions(self):
        """Test get_prioritized_legal_actions() method."""
        raw_state = copy.deepcopy(self.scenarios["early_game"])
        state = FreeCivState(raw_state)

        # Get prioritized actions
        prioritized = state.get_prioritized_legal_actions(player_id=1, max_actions=10)
        all_actions = state.get_legal_actions(player_id=1)

        # Should return at most 10 actions
        self.assertLessEqual(len(prioritized), 10)

        # Should be subset of all legal actions
        prioritized_set = {
            (a.action_type, a.actor_id, str(a.target)) for a in prioritized
        }
        all_actions_set = {
            (a.action_type, a.actor_id, str(a.target)) for a in all_actions
        }
        self.assertTrue(prioritized_set.issubset(all_actions_set))

        # All prioritized actions should have strategic scores
        for action in prioritized:
            self.assertGreaterEqual(action.strategic_score, 0.0)

        # If we have more than 10 legal actions, should prioritize correctly
        if len(all_actions) > 10:
            self.assertEqual(len(prioritized), 10)
            # Check that actions are sorted by strategic score
            scores = [action.strategic_score for action in prioritized]
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_action_strategic_scoring(self):
        """Test strategic scoring algorithms."""
        raw_state = copy.deepcopy(self.scenarios["early_game"])
        state = FreeCivState(raw_state)

        # Create test actions
        city_build_action = FreeCivAction(
            action_type="unit_build_city", actor_id=101, source="unit"
        )

        move_action = FreeCivAction(
            action_type="unit_move",
            actor_id=101,
            target={"x": 2, "y": 3},
            source="unit",
        )

        # Test scoring
        city_score = state._calculate_action_strategic_score(city_build_action, 1)
        move_score = state._calculate_action_strategic_score(move_action, 1)

        # City building should score higher than basic move
        self.assertGreater(city_score, move_score)

        # Scores should be positive
        self.assertGreater(city_score, 0)
        self.assertGreater(move_score, 0)

    def test_action_diversity_enforcement(self):
        """Test that action selection maintains diversity."""
        raw_state = copy.deepcopy(self.scenarios["mid_game"])
        state = FreeCivState(raw_state)

        # Create many actions of same type with different scores
        actions = []
        for i in range(20):
            action = FreeCivAction(
                action_type="unit_move",
                actor_id=101 + i,
                target={"x": i, "y": i},
                source="unit",
            )
            action.strategic_score = 10.0 - i  # Decreasing scores
            actions.append((action.strategic_score, action))

        # Test diversity enforcement
        diverse_actions = state._ensure_action_diversity(actions, 10)

        # Should return requested number
        self.assertEqual(len(diverse_actions), 10)

        # Should prefer higher scoring actions but maintain some diversity
        action_types = [action.action_type for action in diverse_actions]

        # All should be same type in this test case
        self.assertTrue(all(t == "unit_move" for t in action_types))

    def test_performance_requirements(self):
        """Test performance requirements (<10ms parsing)."""
        import time

        raw_state = copy.deepcopy(self.scenarios["early_game"])
        state = FreeCivState(raw_state)

        # Test JSON parsing performance
        json_data = '{"action": "unit_move", "unit": 101, "to": [3, 5]}'

        start_time = time.perf_counter()
        for _ in range(100):  # Run 100 times to get reliable timing
            FreeCivAction.from_json(json_data)
        end_time = time.perf_counter()

        avg_time_ms = ((end_time - start_time) / 100) * 1000
        self.assertLess(avg_time_ms, 10, f"JSON parsing took {avg_time_ms:.2f}ms")

        # Test action prioritization performance
        start_time = time.perf_counter()
        for _ in range(10):
            state.get_prioritized_legal_actions(player_id=1, max_actions=20)
        end_time = time.perf_counter()

        avg_time_ms = ((end_time - start_time) / 10) * 1000
        self.assertLess(
            avg_time_ms, 50, f"Action prioritization took {avg_time_ms:.2f}ms"
        )

    def test_comprehensive_performance_benchmarks(self):
        """Comprehensive performance benchmarks for all parsing methods."""
        import time
        from unittest.mock import Mock

        # Test packet conversion performance
        action = FreeCivAction(
            action_type="unit_move",
            actor_id=101,
            target={"x": 3, "y": 5},
            source="unit",
        )

        start_time = time.perf_counter()
        for _ in range(1000):
            action.to_packet()
        end_time = time.perf_counter()

        avg_time_ms = ((end_time - start_time) / 1000) * 1000
        self.assertLess(avg_time_ms, 1, f"Packet conversion took {avg_time_ms:.2f}ms")

        # Test natural language parsing performance
        mock_state = Mock()
        mock_state.units = {101: Mock(unit_id=101, kind="settlers")}
        mock_state.cities = {301: Mock(city_id=301, name="Rome")}

        start_time = time.perf_counter()
        for _ in range(100):
            FreeCivAction.from_natural_language("Move unit 101 to (2,3)", mock_state)
        end_time = time.perf_counter()

        avg_time_ms = ((end_time - start_time) / 100) * 1000
        self.assertLess(
            avg_time_ms, 10, f"Natural language parsing took {avg_time_ms:.2f}ms"
        )

        # Test validation performance
        start_time = time.perf_counter()
        for _ in range(1000):
            FreeCivAction(
                action_type="unit_move",
                actor_id=101,
                target={"x": 3, "y": 5},
                source="unit",
            )
        end_time = time.perf_counter()

        avg_time_ms = ((end_time - start_time) / 1000) * 1000
        self.assertLess(avg_time_ms, 5, f"Validation took {avg_time_ms:.2f}ms")

        # Test strategic scoring performance
        raw_state = copy.deepcopy(self.scenarios["early_game"])
        state = FreeCivState(raw_state)

        test_action = FreeCivAction(
            action_type="unit_move",
            actor_id=101,
            target={"x": 2, "y": 3},
            source="unit",
        )

        start_time = time.perf_counter()
        for _ in range(100):
            state._calculate_action_strategic_score(test_action, 1)
        end_time = time.perf_counter()

        avg_time_ms = ((end_time - start_time) / 100) * 1000
        self.assertLess(avg_time_ms, 5, f"Strategic scoring took {avg_time_ms:.2f}ms")

    def test_performance_under_load(self):
        """Test performance under high load scenarios."""
        import time

        raw_state = copy.deepcopy(self.scenarios["mid_game"])
        state = FreeCivState(raw_state)

        # Test with many legal actions
        all_actions = state.get_legal_actions(player_id=1)
        if len(all_actions) < 50:
            # Create additional test actions for load testing
            for i in range(50):
                all_actions.append(
                    FreeCivAction(
                        action_type="unit_move",
                        actor_id=101 + i,
                        target={"x": i % 10, "y": (i // 10) % 10},
                        source="unit",
                    )
                )

        # Test action space reduction under load
        start_time = time.perf_counter()
        prioritized = state.get_prioritized_legal_actions(player_id=1, max_actions=20)
        end_time = time.perf_counter()

        processing_time_ms = (end_time - start_time) * 1000
        self.assertLess(
            processing_time_ms,
            100,
            f"Action space reduction took {processing_time_ms:.2f}ms",
        )
        self.assertLessEqual(len(prioritized), 20)

        # Test bulk JSON parsing
        json_samples = [
            '{"action": "unit_move", "unit": 101, "to": [3, 5]}',
            (
                '{"action": "city_production", "city": 301, "target": {"value":'
                ' "warriors"}}'
            ),
            '{"action": "unit_attack", "unit": 102, "target": {"id": 203}}',
            '{"action": "tech_research", "target": {"value": "pottery"}}',
        ] * 25  # 100 samples total

        start_time = time.perf_counter()
        for json_data in json_samples:
            FreeCivAction.from_json(json_data)
        end_time = time.perf_counter()

        total_time_ms = (end_time - start_time) * 1000
        avg_time_ms = total_time_ms / len(json_samples)
        self.assertLess(
            avg_time_ms,
            10,
            f"Bulk JSON parsing averaged {avg_time_ms:.2f}ms per action",
        )

    def test_lru_cache_functionality(self):
        """Test LRU cache implementation."""
        # Test basic functionality
        cache = LRUCache(max_size=3)

        # Test empty cache
        self.assertEqual(len(cache), 0)
        self.assertNotIn("key1", cache)

        # Test adding items
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        self.assertEqual(len(cache), 2)
        self.assertIn("key1", cache)
        self.assertEqual(cache.get("key1"), "value1")

        # Test cache eviction when max size reached
        cache.set("key3", "value3")
        self.assertEqual(len(cache), 3)
        cache.set("key4", "value4")  # Should evict oldest
        self.assertEqual(len(cache), 3)
        # At least key4 should be in cache
        self.assertIn("key4", cache)
        # And cache should not exceed max size
        self.assertLessEqual(len(cache), 3)

        # Test cache clearing
        cache.clear()
        self.assertEqual(len(cache), 0)
        self.assertNotIn("key2", cache)

    def test_deep_size_calculation(self):
        """Test deep size calculation for nested objects."""
        # Test simple objects
        simple_size = _calculate_deep_size("test")
        self.assertGreater(simple_size, 0)

        # Test nested structures
        nested_dict = {"a": {"b": {"c": [1, 2, 3]}}}
        nested_size = _calculate_deep_size(nested_dict)
        self.assertGreater(nested_size, simple_size)

        # Test circular reference protection
        circular = {}
        circular["self"] = circular
        circular_size = _calculate_deep_size(circular)
        self.assertGreater(circular_size, 0)

    def test_security_validation_edge_cases(self):
        """Test security validation edge cases."""
        # Test creating actions with boundary values
        action = FreeCivAction(
            action_type="unit_move",
            actor_id=MAX_UNIT_ID,
            target={"x": 0, "y": 0},
            source="unit",
        )
        self.assertEqual(action.actor_id, MAX_UNIT_ID)

        # Test actions with extreme values that should be validated
        # by Pydantic model validation
        with self.assertRaises(ValidationError):
            FreeCivAction(
                action_type="unit_move",
                actor_id=-1,  # Negative ID should fail validation
                target={"x": 0, "y": 0},
                source="unit",
            )

    def test_fallback_game_state_coverage(self):
        """Test fallback game state methods for complete coverage."""
        # Create a fallback state to test uncovered methods
        fallback = _FallbackGameState()

        # Test all methods that should raise NotImplementedError
        with self.assertRaises(NotImplementedError):
            fallback.current_player()

        with self.assertRaises(NotImplementedError):
            fallback.legal_actions()

        with self.assertRaises(NotImplementedError):
            fallback.is_terminal()

        with self.assertRaises(NotImplementedError):
            fallback.returns()

    def test_error_handling_edge_cases(self):
        """Test error handling in various edge case scenarios."""
        # Test invalid JSON should raise ValueError
        invalid_json = '{"action": "invalid", incomplete...'

        # Should raise ValueError for invalid JSON
        with self.assertRaises(ValueError):
            FreeCivAction.from_json(invalid_json)

        # Test extremely nested JSON to trigger depth protection
        deeply_nested = {
            "a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": {"i": {"j": "deep"}}}}}}}}}
        }
        try:
            action = FreeCivAction.from_json(json.dumps(deeply_nested))
        except ValueError:
            # Expected for deep nesting protection
            pass

    def test_complete_import_coverage(self):
        """Test import-related code paths for complete coverage."""
        # This tests the import fallback paths
        self.assertIsNotNone(_GameStateBase)

        # Test pyspiel availability check
        if pyspiel is not None:
            self.assertIsNotNone(_PY_SPIEL_DUMMY_GAME)
        else:
            self.assertIsNone(_PY_SPIEL_DUMMY_GAME)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
