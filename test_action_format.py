#!/usr/bin/env python3
"""Quick unit test to verify FreeCiv3D action format conversions."""

import sys
sys.path.insert(0, '/app')

from game_arena.harness.freeciv_proxy_client import ProtocolTranslator
from game_arena.harness.freeciv_state import FreeCivAction

def test_tech_research_format():
    """Test tech_research action format conversion."""
    translator = ProtocolTranslator()

    # Create action as it would be stored internally
    action = FreeCivAction(
        action_type="tech_research",
        actor_id=0,  # player_id
        target={"value": "Alphabet"},  # Note: uppercase (should be converted)
        parameters={},
        source="player",
        confidence=1.0,
        parse_method="proxy_flat_list"
    )

    # Convert to gateway format
    packet = translator.to_freeciv_packet(action)

    # Verify format
    expected = {
        "type": "tech_research",
        "tech_name": "alphabet"  # Should be lowercase
    }

    assert packet == expected, f"Expected {expected}, got {packet}"
    print("✅ tech_research format: PASS")
    print(f"   Input: target={action.target}")
    print(f"   Output: {packet}")

def test_unit_move_format():
    """Test unit_move action format conversion."""
    translator = ProtocolTranslator()

    action = FreeCivAction(
        action_type="unit_move",
        actor_id=42,  # unit_id
        target={"x": 15, "y": 20},
        parameters={},
        source="unit",
        confidence=1.0,
        parse_method="proxy_flat_list"
    )

    packet = translator.to_freeciv_packet(action)

    expected = {
        "type": "unit_move",
        "unit_id": 42,
        "dest_x": 15,
        "dest_y": 20
    }

    assert packet == expected, f"Expected {expected}, got {packet}"
    print("✅ unit_move format: PASS")
    print(f"   Input: actor_id={action.actor_id}, target={action.target}")
    print(f"   Output: {packet}")

def test_city_production_format():
    """Test city_production action format conversion."""
    translator = ProtocolTranslator()

    action = FreeCivAction(
        action_type="city_production",
        actor_id=1,  # city_id
        target={"value": "Warrior"},  # Note: uppercase (should be converted)
        parameters={"production_type": "Warrior"},
        source="city",
        confidence=1.0,
        parse_method="proxy_flat_list"
    )

    packet = translator.to_freeciv_packet(action)

    expected = {
        "type": "city_production",
        "city_id": 1,
        "production_type": "warrior"  # Should be lowercase
    }

    assert packet == expected, f"Expected {expected}, got {packet}"
    print("✅ city_production format: PASS")
    print(f"   Input: actor_id={action.actor_id}, parameters={action.parameters}")
    print(f"   Output: {packet}")

def test_unit_fortify_format():
    """Test unit_fortify action format conversion."""
    translator = ProtocolTranslator()

    action = FreeCivAction(
        action_type="unit_fortify",
        actor_id=42,  # unit_id
        target={},
        parameters={},
        source="unit",
        confidence=1.0,
        parse_method="proxy_flat_list"
    )

    packet = translator.to_freeciv_packet(action)

    expected = {
        "type": "unit_fortify",
        "unit_id": 42
    }

    assert packet == expected, f"Expected {expected}, got {packet}"
    print("✅ unit_fortify format: PASS")
    print(f"   Input: actor_id={action.actor_id}")
    print(f"   Output: {packet}")

def test_end_turn_format():
    """Test end_turn action format conversion."""
    translator = ProtocolTranslator()

    action = FreeCivAction(
        action_type="end_turn",
        actor_id=0,  # player_id
        target=None,
        parameters={"turn": 5},  # Should be ignored
        source="player",
        confidence=1.0,
        parse_method="proxy_flat_list"
    )

    packet = translator.to_freeciv_packet(action)

    expected = {
        "type": "end_turn"
    }

    assert packet == expected, f"Expected {expected}, got {packet}"
    print("✅ end_turn format: PASS")
    print(f"   Input: actor_id={action.actor_id}, parameters={action.parameters}")
    print(f"   Output: {packet}")

def test_all_unit_actions():
    """Test all simple unit action formats."""
    translator = ProtocolTranslator()

    unit_actions = [
        "unit_explore",
        "unit_sentry",
        "unit_build_road",
        "unit_build_irrigation",
        "unit_build_mine",
        "unit_build_city"
    ]

    for action_type in unit_actions:
        action = FreeCivAction(
            action_type=action_type,
            actor_id=42,
            target={},
            parameters={},
            source="unit",
            confidence=1.0,
            parse_method="proxy_flat_list"
        )

        packet = translator.to_freeciv_packet(action)

        expected = {
            "type": action_type,
            "unit_id": 42
        }

        assert packet == expected, f"Expected {expected}, got {packet}"
        print(f"✅ {action_type} format: PASS")

if __name__ == "__main__":
    print("=" * 60)
    print("Testing FreeCiv3D Action Format Conversions")
    print("=" * 60)
    print()

    try:
        test_tech_research_format()
        print()
        test_unit_move_format()
        print()
        test_city_production_format()
        print()
        test_unit_fortify_format()
        print()
        test_end_turn_format()
        print()
        test_all_unit_actions()
        print()
        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        sys.exit(0)
    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        sys.exit(1)
    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        sys.exit(1)
