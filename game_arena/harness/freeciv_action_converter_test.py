"""Unit tests for FreeCiv Action Converter.

Tests bidirectional conversion between FreeCivAction objects and their
string representations, ensuring roundtrip consistency.
"""

import unittest
from game_arena.harness.freeciv_action_converter import FreeCivActionConverter
from game_arena.harness.freeciv_state import FreeCivAction, FreeCivState


class TestFreeCivActionConverter(unittest.TestCase):
    """Test cases for FreeCivActionConverter."""

    def setUp(self):
        """Set up test fixtures."""
        self.converter = FreeCivActionConverter()

    def test_parse_unit_explore(self):
        """Test parsing unit_explore action string."""
        action_string = "unit_explore_warrior(101)"

        action = self.converter.string_to_action(action_string)

        self.assertEqual(action.action_type, "unit_explore")
        self.assertEqual(action.actor_id, 101)
        self.assertEqual(action.source, "unit")
        self.assertEqual(action.target, {})

    def test_parse_city_build_improvement(self):
        """Test parsing city_build_improvement action string."""
        action_string = "city_build_improvement_rome(301)_target(barracks)"

        action = self.converter.string_to_action(action_string)

        self.assertEqual(action.action_type, "city_build_improvement")
        self.assertEqual(action.actor_id, 301)
        self.assertEqual(action.source, "city")
        self.assertEqual(action.target, {"value": "barracks"})

    def test_parse_tech_research(self):
        """Test parsing tech_research action string."""
        action_string = "tech_research_player(1)_target(Alphabet)"

        action = self.converter.string_to_action(action_string)

        self.assertEqual(action.action_type, "tech_research")
        self.assertEqual(action.actor_id, 1)
        self.assertEqual(action.source, "player")
        self.assertEqual(action.target, {"value": "Alphabet"})

    def test_parse_tech_research_with_spaces(self):
        """Test parsing tech_research action with tech name containing spaces."""
        action_string = "tech_research_player(1)_target(Bronze Working)"

        action = self.converter.string_to_action(action_string)

        self.assertEqual(action.action_type, "tech_research")
        self.assertEqual(action.actor_id, 1)
        self.assertEqual(action.target, {"value": "Bronze Working"})

    def test_roundtrip_unit_explore(self):
        """Test roundtrip conversion for unit_explore action."""
        original_action = FreeCivAction(
            action_type="unit_explore",
            actor_id=101,
            target={},
            parameters={},
            source="unit"
        )

        # Convert to string
        action_string = self.converter.action_to_string(original_action)
        self.assertEqual(action_string, "unit_explore_unit(101)")

        # Convert back to action
        parsed_action = self.converter.string_to_action(action_string)

        self.assertEqual(parsed_action.action_type, original_action.action_type)
        self.assertEqual(parsed_action.actor_id, original_action.actor_id)
        self.assertEqual(parsed_action.source, original_action.source)

    def test_roundtrip_city_build_improvement(self):
        """Test roundtrip conversion for city_build_improvement action."""
        original_action = FreeCivAction(
            action_type="city_build_improvement",
            actor_id=301,
            target={"value": "barracks"},
            parameters={},
            source="city"
        )

        # Convert to string
        action_string = self.converter.action_to_string(original_action)
        self.assertEqual(action_string, "city_build_improvement_city(301)_target(barracks)")

        # Convert back to action
        parsed_action = self.converter.string_to_action(action_string)

        self.assertEqual(parsed_action.action_type, original_action.action_type)
        self.assertEqual(parsed_action.actor_id, original_action.actor_id)
        self.assertEqual(parsed_action.source, original_action.source)
        self.assertEqual(parsed_action.target, original_action.target)

    def test_roundtrip_tech_research(self):
        """Test roundtrip conversion for tech_research action."""
        original_action = FreeCivAction(
            action_type="tech_research",
            actor_id=1,
            target={"value": "Alphabet"},
            parameters={},
            source="player"
        )

        # Convert to string
        action_string = self.converter.action_to_string(original_action)
        self.assertEqual(action_string, "tech_research_player(1)_target(Alphabet)")

        # Convert back to action
        parsed_action = self.converter.string_to_action(action_string)

        self.assertEqual(parsed_action.action_type, original_action.action_type)
        self.assertEqual(parsed_action.actor_id, original_action.actor_id)
        self.assertEqual(parsed_action.source, original_action.source)
        self.assertEqual(parsed_action.target, original_action.target)

    def test_parse_existing_actions_still_work(self):
        """Test that existing action parsers still work correctly."""
        # Test unit_move
        move_action = self.converter.string_to_action("unit_move_warrior(101)_to(5,10)")
        self.assertEqual(move_action.action_type, "unit_move")
        self.assertEqual(move_action.actor_id, 101)
        self.assertEqual(move_action.target, {"x": 5, "y": 10})

        # Test unit_attack
        attack_action = self.converter.string_to_action("unit_attack_warrior(101)_target(202)")
        self.assertEqual(attack_action.action_type, "unit_attack")
        self.assertEqual(attack_action.actor_id, 101)
        self.assertEqual(attack_action.target, {"id": 202})

        # Test unit_fortify
        fortify_action = self.converter.string_to_action("unit_fortify_warrior(101)")
        self.assertEqual(fortify_action.action_type, "unit_fortify")
        self.assertEqual(fortify_action.actor_id, 101)

        # Test city_production
        prod_action = self.converter.string_to_action("city_production_rome(301)_target(warriors)")
        self.assertEqual(prod_action.action_type, "city_production")
        self.assertEqual(prod_action.actor_id, 301)
        self.assertEqual(prod_action.target, {"value": "warriors"})


class TestProtocolTranslator(unittest.TestCase):
    """Test cases for ProtocolTranslator."""

    def setUp(self):
        """Set up test fixtures."""
        from game_arena.harness.freeciv_proxy_client import ProtocolTranslator
        self.translator = ProtocolTranslator()

    def test_translate_unit_explore(self):
        """Test protocol translation for unit_explore action."""
        action = FreeCivAction(
            action_type="unit_explore",
            actor_id=101,
            target={},
            parameters={},
            source="unit"
        )

        packet = self.translator.to_freeciv_packet(action)

        self.assertEqual(packet["action_type"], "unit_explore")
        self.assertEqual(packet["actor_id"], 101)
        self.assertEqual(packet["target"], {})

    def test_translate_unit_attack(self):
        """Test protocol translation for unit_attack action."""
        action = FreeCivAction(
            action_type="unit_attack",
            actor_id=101,
            target={"id": 202},
            parameters={},
            source="unit"
        )

        packet = self.translator.to_freeciv_packet(action)

        self.assertEqual(packet["action_type"], "unit_attack")
        self.assertEqual(packet["actor_id"], 101)
        self.assertEqual(packet["target"], {"id": 202})

    def test_translate_unit_fortify(self):
        """Test protocol translation for unit_fortify action."""
        action = FreeCivAction(
            action_type="unit_fortify",
            actor_id=101,
            target={},
            parameters={},
            source="unit"
        )

        packet = self.translator.to_freeciv_packet(action)

        self.assertEqual(packet["action_type"], "unit_fortify")
        self.assertEqual(packet["actor_id"], 101)

    def test_translate_unit_build_city(self):
        """Test protocol translation for unit_build_city action."""
        action = FreeCivAction(
            action_type="unit_build_city",
            actor_id=101,
            target={},
            parameters={},
            source="unit"
        )

        packet = self.translator.to_freeciv_packet(action)

        self.assertEqual(packet["action_type"], "unit_build_city")
        self.assertEqual(packet["actor_id"], 101)

    def test_translate_city_build_improvement(self):
        """Test protocol translation for city_build_improvement action."""
        action = FreeCivAction(
            action_type="city_build_improvement",
            actor_id=301,
            target={"value": "barracks"},
            parameters={},
            source="city"
        )

        packet = self.translator.to_freeciv_packet(action)

        self.assertEqual(packet["action_type"], "city_build_improvement")
        self.assertEqual(packet["actor_id"], 301)
        self.assertEqual(packet["target"], {"value": "barracks"})

    def test_translate_tech_research(self):
        """Test protocol translation for tech_research action."""
        action = FreeCivAction(
            action_type="tech_research",
            actor_id=1,
            target={"value": "Alphabet"},
            parameters={},
            source="player"
        )

        packet = self.translator.to_freeciv_packet(action)

        self.assertEqual(packet["action_type"], "tech_research")
        self.assertEqual(packet["actor_id"], 1)
        self.assertEqual(packet["target"], {"value": "Alphabet"})

    def test_translate_existing_actions_still_work(self):
        """Test that existing action translations still work correctly."""
        # Test unit_move
        move_action = FreeCivAction(
            action_type="unit_move",
            actor_id=101,
            target={"x": 5, "y": 10},
            parameters={},
            source="unit"
        )
        move_packet = self.translator.to_freeciv_packet(move_action)
        self.assertEqual(move_packet["action_type"], "unit_move")
        self.assertEqual(move_packet["actor_id"], 101)
        self.assertEqual(move_packet["target"], {"x": 5, "y": 10})

        # Test city_production
        prod_action = FreeCivAction(
            action_type="city_production",
            actor_id=301,
            target={"value": "warriors"},
            parameters={},
            source="city"
        )
        prod_packet = self.translator.to_freeciv_packet(prod_action)
        self.assertEqual(prod_packet["action_type"], "city_production")
        self.assertEqual(prod_packet["actor_id"], 301)
        self.assertEqual(prod_packet["target"], {"value": "warriors"})

    def test_full_roundtrip_tech_research(self):
        """Test full roundtrip: string → action → packet → verify proxy compatibility."""
        converter = FreeCivActionConverter()

        # Start with action string that LLM generates
        action_string = "tech_research_player(1)_target(Bronze Working)"

        # Parse to action
        action = converter.string_to_action(action_string)
        self.assertEqual(action.action_type, "tech_research")

        # Translate to proxy packet
        packet = self.translator.to_freeciv_packet(action)

        # Verify proxy-compatible format (action_type + actor_id + target structure)
        self.assertEqual(packet["action_type"], "tech_research")
        self.assertEqual(packet["actor_id"], 1)
        self.assertEqual(packet["target"], {"value": "Bronze Working"})

        # Verify NO nested 'data' field (old bug)
        self.assertNotIn("data", packet)


class TestProxyActionConversion(unittest.TestCase):
    """Test cases for converting proxy actions to FreeCivAction objects."""

    def test_convert_tech_research_from_proxy_format(self):
        """Test that tech_research actions from proxy are converted correctly.

        The proxy sends tech_research actions with 'tech_name' field,
        not 'target' field. This test verifies the special handling works.
        """
        from game_arena.harness.freeciv_state import FreeCivState

        # Simulate proxy action format for tech_research
        proxy_action = {
            "type": "tech_research",
            "tech_name": "Bronze Working",
            "priority": "high"
        }

        # Create minimal state for testing
        raw_state = {
            "game": {"turn": 1, "phase": "movement"},
            "player_id": 0,
            "units": [],
            "cities": [],
            "players": [{"id": 0, "name": "test"}],
            "map": {"width": 64, "height": 64, "tiles": []},
        }
        state = FreeCivState(raw_state)

        # Convert proxy action using _convert_action
        action = state._convert_action(
            actor_id=0,  # player_id
            action_data=proxy_action,
            source="player"
        )

        # Verify conversion
        self.assertEqual(action.action_type, "tech_research")
        self.assertEqual(action.actor_id, 0)
        self.assertEqual(action.source, "player")
        self.assertIsNotNone(action.target)
        self.assertEqual(action.target.get("value"), "Bronze Working")

    def test_convert_tech_research_roundtrip_from_proxy(self):
        """Test full roundtrip: proxy format → FreeCivAction → string → parse."""
        from game_arena.harness.freeciv_state import FreeCivState
        from game_arena.harness.freeciv_action_converter import FreeCivActionConverter

        # Simulate proxy action
        proxy_action = {
            "type": "tech_research",
            "tech_name": "Alphabet",
            "priority": "high"
        }

        # Create state and converter
        raw_state = {
            "game": {"turn": 1, "phase": "movement"},
            "player_id": 0,
            "units": [],
            "cities": [],
            "players": [{"id": 0, "name": "test"}],
            "map": {"width": 64, "height": 64, "tiles": []},
        }
        state = FreeCivState(raw_state)
        converter = FreeCivActionConverter()

        # Convert from proxy format
        action = state._convert_action(
            actor_id=0,
            action_data=proxy_action,
            source="player"
        )

        # Convert to string
        action_string = converter.action_to_string(action)
        self.assertEqual(action_string, "tech_research_player(0)_target(Alphabet)")

        # Parse back
        parsed_action = converter.string_to_action(action_string)
        self.assertEqual(parsed_action.action_type, "tech_research")
        self.assertEqual(parsed_action.actor_id, 0)
        self.assertEqual(parsed_action.target.get("value"), "Alphabet")

    def test_parse_proxy_action_tech_research(self):
        """Test that _parse_proxy_action handles tech_research correctly.

        This is critical for legal_actions which uses _parse_proxy_action.
        """
        from game_arena.harness.freeciv_state import FreeCivState

        # Simulate proxy legal action for tech_research
        proxy_action = {
            "type": "tech_research",
            "tech_name": "Writing",
            "priority": "high",
            "description": "Research Writing technology"
        }

        # Create state
        raw_state = {
            "game": {"turn": 1, "phase": "movement"},
            "player_id": 0,
            "units": [],
            "cities": [],
            "players": [{"id": 0, "name": "test"}],
            "map": {"width": 64, "height": 64, "tiles": []},
        }
        state = FreeCivState(raw_state)

        # Parse using _parse_proxy_action (used by get_legal_actions)
        action = state._parse_proxy_action(
            action_data=proxy_action,
            actor_id=0
        )

        # Verify target is set correctly (not None!)
        self.assertEqual(action.action_type, "tech_research")
        self.assertEqual(action.actor_id, 0)
        self.assertEqual(action.source, "player")
        self.assertIsNotNone(action.target, "target must not be None for protocol translator!")
        self.assertEqual(action.target.get("value"), "Writing")

        # Verify tech_name is NOT in parameters (it's in target now)
        self.assertNotIn("tech_name", action.parameters)

    def test_parse_proxy_action_tech_research_protocol_translation(self):
        """Test that tech_research from _parse_proxy_action can be sent via protocol translator."""
        from game_arena.harness.freeciv_state import FreeCivState
        from game_arena.harness.freeciv_proxy_client import ProtocolTranslator

        # Simulate proxy legal action
        proxy_action = {
            "type": "tech_research",
            "tech_name": "Mathematics",
            "priority": "medium"
        }

        # Create state and translator
        raw_state = {
            "game": {"turn": 1, "phase": "movement"},
            "player_id": 0,
            "units": [],
            "cities": [],
            "players": [{"id": 0, "name": "test"}],
            "map": {"width": 64, "height": 64, "tiles": []},
        }
        state = FreeCivState(raw_state)
        translator = ProtocolTranslator()

        # Parse action
        action = state._parse_proxy_action(proxy_action, actor_id=0)

        # This should NOT crash with AttributeError: 'NoneType' object has no attribute 'get'
        packet = translator.to_freeciv_packet(action)

        # Verify packet format (action_type + actor_id + target structure)
        self.assertEqual(packet["action_type"], "tech_research")
        self.assertEqual(packet["actor_id"], 0)
        self.assertEqual(packet["target"], {"value": "Mathematics"})


if __name__ == "__main__":
    unittest.main()
