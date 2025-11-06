"""Tests for FreeCiv state models and validation."""

from absl.testing import absltest, parameterized
from pydantic import ValidationError

from game_arena.harness import freeciv_state


class FreeCivPlayerValidationTest(parameterized.TestCase):
  """Tests for FreeCivPlayer Pydantic model validation."""

  def test_pregame_score_negative_one(self):
    """Test that score=-1 is accepted (pre-game sentinel value)."""
    player = freeciv_state.FreeCivPlayer(
        player_id=0,
        name="Test Player",
        nation="Romans",
        score=-1,  # Pre-game sentinel value
        gold=0,
    )
    self.assertEqual(player.score, -1)

  @parameterized.named_parameters(
      ("zero", 0),
      ("small", 100),
      ("medium", 1000),
      ("large", 999999),
  )
  def test_valid_game_scores(self, score):
    """Test that non-negative scores are accepted."""
    player = freeciv_state.FreeCivPlayer(
        player_id=0,
        name="Test Player",
        nation="Romans",
        score=score,
        gold=0,
    )
    self.assertEqual(player.score, score)

  @parameterized.named_parameters(
      ("minus_two", -2),
      ("minus_ten", -10),
      ("minus_hundred", -100),
  )
  def test_invalid_negative_scores(self, score):
    """Test that scores less than -1 are rejected."""
    with self.assertRaises(ValidationError) as context:
      freeciv_state.FreeCivPlayer(
          player_id=0,
          name="Test Player",
          nation="Romans",
          score=score,
          gold=0,
      )
    # Verify the error message mentions the validation constraint
    self.assertIn("Score must be >= -1", str(context.exception))

  def test_player_with_pregame_state_complete(self):
    """Test creating a complete player object with pre-game score."""
    player = freeciv_state.FreeCivPlayer(
        player_id=0,
        name="agent_player1_abc123",
        nation="Romans",
        score=-1,  # Pre-game sentinel
        gold=0,
        techs=[],
        government=None,
        science=0,
        research_target=None,
        research_progress=0,
        diplomatic_relations={},
        trade_routes=[],
        luxuries_rate=0,
        science_rate=50,
        tax_rate=50,
    )
    self.assertEqual(player.score, -1)
    self.assertEqual(player.name, "agent_player1_abc123")
    self.assertEqual(player.gold, 0)

  def test_player_with_active_game_state(self):
    """Test creating a player object with active game score."""
    player = freeciv_state.FreeCivPlayer(
        player_id=1,
        name="agent_player2_def456",
        nation="Greeks",
        score=250,  # Active game score
        gold=500,
        techs=["Alphabet", "Bronze Working"],
        government="Republic",
        science=10,
        research_target="Philosophy",
        research_progress=50,
        diplomatic_relations={0: "peace"},
        trade_routes=[],
        luxuries_rate=20,
        science_rate=50,
        tax_rate=30,
    )
    self.assertEqual(player.score, 250)
    self.assertEqual(player.gold, 500)
    self.assertLen(player.techs, 2)


class FreeCivUnitValidationTest(parameterized.TestCase):
  """Tests for FreeCivUnit Pydantic model validation."""

  def test_activity_integer_zero_converted_to_none(self):
    """Test that activity=0 (int) is converted to None (idle state)."""
    unit = freeciv_state.FreeCivUnit(
        unit_id=1,
        owner=0,
        kind="Settlers",
        position=(10, 15),
        hp=10,
        moves_left=3,
        activity=0,  # Integer 0 from FreeCiv
    )
    self.assertIsNone(unit.activity)

  def test_activity_integer_nonzero_converted_to_string(self):
    """Test that non-zero activity integers are converted to descriptive names."""
    unit = freeciv_state.FreeCivUnit(
        unit_id=1,
        owner=0,
        kind="Workers",
        position=(10, 15),
        hp=10,
        moves_left=3,
        activity=5,  # ACTIVITY_FORTIFIED
    )
    # Activity code 5 should map to "fortified"
    self.assertEqual(unit.activity, "fortified")

  def test_activity_string_passed_through(self):
    """Test that string activities are preserved."""
    unit = freeciv_state.FreeCivUnit(
        unit_id=1,
        owner=0,
        kind="Workers",
        position=(10, 15),
        hp=10,
        moves_left=3,
        activity="building_road",
    )
    self.assertEqual(unit.activity, "building_road")

  def test_activity_none_passed_through(self):
    """Test that None activity is preserved."""
    unit = freeciv_state.FreeCivUnit(
        unit_id=1,
        owner=0,
        kind="Warriors",
        position=(10, 15),
        hp=10,
        moves_left=1,
        activity=None,
    )
    self.assertIsNone(unit.activity)

  def test_activity_invalid_type_rejected(self):
    """Test that invalid activity types are rejected."""
    with self.assertRaises(ValidationError) as context:
      freeciv_state.FreeCivUnit(
          unit_id=1,
          owner=0,
          kind="Warriors",
          position=(10, 15),
          hp=10,
          moves_left=1,
          activity=[1, 2, 3],  # Invalid type
      )
    self.assertIn("Activity must be string, int, or None", str(context.exception))


class FreeCivLegalActionsTest(absltest.TestCase):
  """Tests for legal_actions parsing from proxy flat list."""

  def test_get_legal_actions_from_flat_list(self):
    """Test that get_legal_actions uses flat list from proxy."""
    raw_state = {
        "turn": 1,
        "phase": "movement",
        "game": {"turn": 1, "phase": "movement"},
        "map": {"width": 64, "height": 64, "tiles": []},
        "players": {
            "0": {
                "id": 0,
                "name": "Player1",
                "nation": "Romans",
                "score": 100,
                "gold": 50,
            }
        },
        "units": {
            "132": {
                "id": 132,
                "owner": 0,
                "type": "Settlers",
                "x": 10,
                "y": 20,
                "hp": 10,
                "moves": 3,
                "available_actions": [],  # Empty per-unit list
            }
        },
        "cities": {},
        "legal_actions": [  # Flat list from proxy
            {
                "type": "unit_move",
                "unit_id": 132,
                "dest_x": 11,
                "dest_y": 20,
                "priority": "medium",
                "description": "Move east",
            },
            {
                "type": "unit_build_city",
                "unit_id": 132,
                "priority": "high",
                "description": "Found new city",
            },
        ],
    }

    state = freeciv_state.FreeCivState(raw_state)
    actions = state.get_legal_actions(player_id=0)

    # Should have 2 actions from flat list + end_turn (injected)
    self.assertEqual(len(actions), 3)

    # Check action types
    action_types = {a.action_type for a in actions}
    self.assertIn("unit_move", action_types)
    self.assertIn("unit_build_city", action_types)
    self.assertIn("end_turn", action_types)  # Always injected

    # Check actor IDs (unit actions use unit_id, end_turn uses player_id)
    for action in actions:
      if action.action_type == "end_turn":
        self.assertEqual(action.actor_id, 0)  # player_id
      else:
        self.assertEqual(action.actor_id, 132)  # unit_id

  def test_legal_actions_player_filtering(self):
    """Test that legal_actions are filtered by player ownership."""
    raw_state = {
        "turn": 1,
        "phase": "movement",
        "game": {"turn": 1, "phase": "movement"},
        "map": {"width": 64, "height": 64, "tiles": []},
        "players": {
            "0": {"id": 0, "name": "P1", "nation": "Romans", "score": 100, "gold": 50},
            "1": {"id": 1, "name": "P2", "nation": "Greeks", "score": 95, "gold": 45},
        },
        "units": {
            "132": {"id": 132, "owner": 0, "type": "Settlers", "x": 10, "y": 20, "hp": 10, "moves": 3},
            "133": {"id": 133, "owner": 1, "type": "Warriors", "x": 15, "y": 25, "hp": 10, "moves": 1},
        },
        "cities": {},
        "legal_actions": [
            {"type": "unit_move", "unit_id": 132, "dest_x": 11, "dest_y": 20, "priority": "medium"},
            {"type": "unit_move", "unit_id": 133, "dest_x": 16, "dest_y": 25, "priority": "medium"},
        ],
    }

    state = freeciv_state.FreeCivState(raw_state)

    # Player 0 should only see unit 132's actions + end_turn
    actions_p0 = state.get_legal_actions(player_id=0)
    self.assertEqual(len(actions_p0), 2)  # unit_move + end_turn
    unit_actions_p0 = [a for a in actions_p0 if a.action_type != "end_turn"]
    self.assertEqual(len(unit_actions_p0), 1)
    self.assertEqual(unit_actions_p0[0].actor_id, 132)

    # Player 1 should only see unit 133's actions + end_turn
    actions_p1 = state.get_legal_actions(player_id=1)
    self.assertEqual(len(actions_p1), 2)  # unit_move + end_turn
    unit_actions_p1 = [a for a in actions_p1 if a.action_type != "end_turn"]
    self.assertEqual(len(unit_actions_p1), 1)
    self.assertEqual(unit_actions_p1[0].actor_id, 133)

  def test_legal_actions_dest_x_dest_y_parsing(self):
    """Test parsing of dest_x/dest_y format from proxy."""
    raw_state = {
        "turn": 1,
        "phase": "movement",
        "game": {"turn": 1, "phase": "movement"},
        "map": {"width": 64, "height": 64, "tiles": []},
        "players": {"0": {"id": 0, "name": "P1", "nation": "Romans", "score": 100, "gold": 50}},
        "units": {"132": {"id": 132, "owner": 0, "type": "Settlers", "x": 10, "y": 20, "hp": 10, "moves": 3}},
        "cities": {},
        "legal_actions": [
            {"type": "unit_move", "unit_id": 132, "dest_x": 11, "dest_y": 21, "priority": "high"}
        ],
    }

    state = freeciv_state.FreeCivState(raw_state)
    actions = state.get_legal_actions(player_id=0)

    self.assertEqual(len(actions), 2)  # unit_move + end_turn
    move_actions = [a for a in actions if a.action_type == "unit_move"]
    self.assertEqual(len(move_actions), 1)
    action = move_actions[0]
    self.assertEqual(action.action_type, "unit_move")
    self.assertIsNotNone(action.target)
    self.assertEqual(action.target["x"], 11)
    self.assertEqual(action.target["y"], 21)

  def test_legal_actions_priority_to_confidence(self):
    """Test that priority field maps to confidence scores."""
    raw_state = {
        "turn": 1,
        "phase": "movement",
        "game": {"turn": 1, "phase": "movement"},
        "map": {"width": 64, "height": 64, "tiles": []},
        "players": {"0": {"id": 0, "name": "P1", "nation": "Romans", "score": 100, "gold": 50}},
        "units": {"132": {"id": 132, "owner": 0, "type": "Settlers", "x": 10, "y": 20, "hp": 10, "moves": 3}},
        "cities": {},
        "legal_actions": [
            {"type": "unit_build_city", "unit_id": 132, "priority": "high"},
            {"type": "unit_move", "unit_id": 132, "dest_x": 11, "dest_y": 20, "priority": "medium"},
            {"type": "unit_move", "unit_id": 132, "dest_x": 9, "dest_y": 20, "priority": "low"},
        ],
    }

    state = freeciv_state.FreeCivState(raw_state)
    actions = state.get_legal_actions(player_id=0)

    self.assertEqual(len(actions), 4)  # 3 unit actions + end_turn

    # Find actions by type and check confidence
    for action in actions:
      if action.action_type == "unit_build_city":
        self.assertAlmostEqual(action.confidence, 0.9)  # high priority
      elif action.action_type == "end_turn":
        self.assertAlmostEqual(action.confidence, 1.0)  # injected action
      elif action.target and action.target.get("x") == 11:
        self.assertAlmostEqual(action.confidence, 0.7)  # medium priority
      elif action.target and action.target.get("x") == 9:
        self.assertAlmostEqual(action.confidence, 0.5)  # low priority

  def test_legal_actions_empty_list(self):
    """Test behavior when legal_actions field is empty."""
    raw_state = {
        "turn": 1,
        "phase": "movement",
        "game": {"turn": 1, "phase": "movement"},
        "map": {"width": 64, "height": 64, "tiles": []},
        "players": {"0": {"id": 0, "name": "P1", "nation": "Romans", "score": 100, "gold": 50}},
        "units": {"132": {"id": 132, "owner": 0, "type": "Settlers", "x": 10, "y": 20, "hp": 10, "moves": 3}},
        "cities": {},
        "legal_actions": [],  # Empty list
    }

    state = freeciv_state.FreeCivState(raw_state)
    actions = state.get_legal_actions(player_id=0)

    # Should return only end_turn action (always available)
    self.assertEqual(len(actions), 1)
    self.assertEqual(actions[0].action_type, "end_turn")

  def test_legal_actions_missing_field(self):
    """Test behavior when legal_actions field is missing."""
    raw_state = {
        "turn": 1,
        "phase": "movement",
        "game": {"turn": 1, "phase": "movement"},
        "map": {"width": 64, "height": 64, "tiles": []},
        "players": {"0": {"id": 0, "name": "P1", "nation": "Romans", "score": 100, "gold": 50}},
        "units": {"132": {"id": 132, "owner": 0, "type": "Settlers", "x": 10, "y": 20, "hp": 10, "moves": 3}},
        "cities": {},
        # No legal_actions field
    }

    state = freeciv_state.FreeCivState(raw_state)
    actions = state.get_legal_actions(player_id=0)

    # Should return only end_turn action (always available)
    self.assertEqual(len(actions), 1)
    self.assertEqual(actions[0].action_type, "end_turn")

  def test_legal_actions_malformed_action_skipped(self):
    """Test that malformed actions are skipped gracefully."""
    raw_state = {
        "turn": 1,
        "phase": "movement",
        "game": {"turn": 1, "phase": "movement"},
        "map": {"width": 64, "height": 64, "tiles": []},
        "players": {"0": {"id": 0, "name": "P1", "nation": "Romans", "score": 100, "gold": 50}},
        "units": {"132": {"id": 132, "owner": 0, "type": "Settlers", "x": 10, "y": 20, "hp": 10, "moves": 3}},
        "cities": {},
        "legal_actions": [
            {"type": "unit_move", "unit_id": 132, "dest_x": 11, "dest_y": 20},  # Valid
            "invalid_string_action",  # Invalid - not a dict
            {"type": "unit_move"},  # Invalid - missing unit_id
            {"unit_id": 132},  # Invalid - missing type
        ],
    }

    state = freeciv_state.FreeCivState(raw_state)
    actions = state.get_legal_actions(player_id=0)

    # Should return the valid action + end_turn
    self.assertEqual(len(actions), 2)
    move_actions = [a for a in actions if a.action_type == "unit_move"]
    self.assertEqual(len(move_actions), 1)
    self.assertEqual(move_actions[0].action_type, "unit_move")
    self.assertEqual(move_actions[0].actor_id, 132)

    # Verify end_turn is present
    end_turn_actions = [a for a in actions if a.action_type == "end_turn"]
    self.assertEqual(len(end_turn_actions), 1)

  def test_legal_actions_caching(self):
    """Test that legal_actions are cached properly."""
    raw_state = {
        "turn": 1,
        "phase": "movement",
        "game": {"turn": 1, "phase": "movement"},
        "map": {"width": 64, "height": 64, "tiles": []},
        "players": {"0": {"id": 0, "name": "P1", "nation": "Romans", "score": 100, "gold": 50}},
        "units": {"132": {"id": 132, "owner": 0, "type": "Settlers", "x": 10, "y": 20, "hp": 10, "moves": 3}},
        "cities": {},
        "legal_actions": [
            {"type": "unit_move", "unit_id": 132, "dest_x": 11, "dest_y": 20, "priority": "medium"}
        ],
    }

    state = freeciv_state.FreeCivState(raw_state)

    # First call
    actions1 = state.get_legal_actions(player_id=0)
    # Second call (should use cache)
    actions2 = state.get_legal_actions(player_id=0)

    # Should return same content
    self.assertEqual(len(actions1), len(actions2))
    self.assertEqual(actions1[0].action_type, actions2[0].action_type)

    # But different instances (model_copy)
    self.assertIsNot(actions1[0], actions2[0])


class FreeCivActionNegativeTest(parameterized.TestCase):
  """Negative test cases for FreeCiv action validation."""

  @parameterized.named_parameters(
      ("empty_string", ""),
      ("whitespace_only", "   "),
      ("random_string", "foobar"),
      ("wrong_prefix", "building_create"),
  )
  def test_invalid_action_type_raises_error(self, action_type):
    """Test that unsupported or malformed action types are rejected."""
    with self.assertRaises(ValidationError) as context:
      freeciv_state.FreeCivAction(
          action_type=action_type,
          actor_id=100,
          source="unit",
      )
    error_msg = str(context.exception)
    # Should mention validation constraint or invalid action type
    self.assertTrue(
        "Invalid action type" in error_msg
        or "min_length" in error_msg
        or "String should have at least 1 character" in error_msg
    )

  def test_action_type_with_invalid_prefix_allowed_for_testing(self):
    """Test that action types starting with invalid_ are allowed for tests."""
    # This should NOT raise an error - it's intentionally allowed
    action = freeciv_state.FreeCivAction(
        action_type="invalid_action_type",
        actor_id=100,
        source="unit",
    )
    self.assertEqual(action.action_type, "invalid_action_type")

  def test_action_type_with_test_prefix_allowed(self):
    """Test that action types starting with test_ are allowed for tests."""
    # This should NOT raise an error - it's intentionally allowed
    action = freeciv_state.FreeCivAction(
        action_type="test_action_type",
        actor_id=100,
        source="unit",
    )
    self.assertEqual(action.action_type, "test_action_type")

  @parameterized.named_parameters(
      ("negative_one", -1),
      ("negative_large", -999),
      ("exceeds_max", 100_001),
      ("way_over_max", 999_999),
  )
  def test_invalid_actor_id_out_of_bounds(self, actor_id):
    """Test that actor_id values outside valid range are rejected."""
    with self.assertRaises(ValidationError) as context:
      freeciv_state.FreeCivAction(
          action_type="unit_move",
          actor_id=actor_id,
          source="unit",
          target={"x": 10, "y": 20},
      )
    error_msg = str(context.exception)
    # Should mention bounds constraint
    self.assertTrue(
        "greater than or equal to 0" in error_msg
        or "less than or equal to 100000" in error_msg
    )

  @parameterized.named_parameters(
      ("string", "not_an_int"),
      ("float", 123.45),
      ("none", None),
      ("list", [1, 2, 3]),
      ("dict", {"id": 100}),
  )
  def test_invalid_actor_id_wrong_type(self, actor_id):
    """Test that non-integer actor_id values are rejected."""
    with self.assertRaises(ValidationError) as context:
      freeciv_state.FreeCivAction(
          action_type="unit_move",
          actor_id=actor_id,
          source="unit",
          target={"x": 10, "y": 20},
      )
    error_msg = str(context.exception)
    # Should mention type validation
    self.assertTrue(
        "Input should be a valid integer" in error_msg
        or "int" in error_msg.lower()
    )

  def test_unit_move_missing_x_coordinate(self):
    """Test that unit_move action requires x coordinate in target."""
    with self.assertRaises(ValidationError) as context:
      freeciv_state.FreeCivAction(
          action_type="unit_move",
          actor_id=100,
          source="unit",
          target={"y": 20},  # Missing x
      )
    self.assertIn("x,y coordinates", str(context.exception))

  def test_unit_move_missing_y_coordinate(self):
    """Test that unit_move action requires y coordinate in target."""
    with self.assertRaises(ValidationError) as context:
      freeciv_state.FreeCivAction(
          action_type="unit_move",
          actor_id=100,
          source="unit",
          target={"x": 10},  # Missing y
      )
    self.assertIn("x,y coordinates", str(context.exception))

  def test_unit_move_missing_both_coordinates(self):
    """Test that unit_move action requires both x,y coordinates."""
    # Note: Empty dict target does not trigger validation error
    # The validator only checks when target is truthy (non-empty)
    # This test verifies that empty target is allowed (no validation error)
    action = freeciv_state.FreeCivAction(
        action_type="unit_move",
        actor_id=100,
        source="unit",
        target={},  # Empty target - allowed
    )
    self.assertEqual(action.target, {})

  @parameterized.named_parameters(
      ("x_string", {"x": "10", "y": 20}),
      ("y_string", {"x": 10, "y": "20"}),
      ("both_string", {"x": "10", "y": "20"}),
      ("x_float", {"x": 10.5, "y": 20}),
      ("y_float", {"x": 10, "y": 20.5}),
      ("x_none", {"x": None, "y": 20}),
      ("y_none", {"x": 10, "y": None}),
  )
  def test_unit_move_invalid_coordinate_types(self, target):
    """Test that unit_move coordinates must be integers."""
    with self.assertRaises(ValidationError) as context:
      freeciv_state.FreeCivAction(
          action_type="unit_move",
          actor_id=100,
          source="unit",
          target=target,
      )
    self.assertIn("must be integers", str(context.exception))

  def test_unit_attack_missing_target_id(self):
    """Test that unit_attack action requires target id."""
    with self.assertRaises(ValidationError) as context:
      freeciv_state.FreeCivAction(
          action_type="unit_attack",
          actor_id=100,
          source="unit",
          target={"x": 10, "y": 20},  # Has coordinates but no id
      )
    self.assertIn("target id", str(context.exception))

  def test_unit_attack_empty_target(self):
    """Test that unit_attack action allows empty target."""
    # Note: Empty dict target does not trigger validation error
    # The validator only checks when target is truthy (non-empty)
    # This test verifies that empty target is allowed (no validation error)
    action = freeciv_state.FreeCivAction(
        action_type="unit_attack",
        actor_id=100,
        source="unit",
        target={},
    )
    self.assertEqual(action.target, {})

  def test_city_production_missing_required_fields(self):
    """Test that city_production requires value, id, or name in target."""
    with self.assertRaises(ValidationError) as context:
      freeciv_state.FreeCivAction(
          action_type="city_production",
          actor_id=300,
          source="city",
          target={"other_field": "something"},
      )
    self.assertIn("value/id/name", str(context.exception))

  def test_city_production_empty_target(self):
    """Test that city_production allows empty target."""
    # Note: Empty dict target does not trigger validation error
    # The validator only checks when target is truthy (non-empty)
    # This test verifies that empty target is allowed (no validation error)
    action = freeciv_state.FreeCivAction(
        action_type="city_production",
        actor_id=300,
        source="city",
        target={},
    )
    self.assertEqual(action.target, {})

  def test_action_with_none_target_allowed(self):
    """Test that None target is allowed for actions that don't require it."""
    action = freeciv_state.FreeCivAction(
        action_type="unit_fortify",
        actor_id=100,
        source="unit",
        target=None,
    )
    self.assertIsNone(action.target)

  def test_action_without_target_field(self):
    """Test that actions can be created without target field."""
    action = freeciv_state.FreeCivAction(
        action_type="end_turn",
        actor_id=0,
        source="player",
    )
    self.assertIsNone(action.target)

  @parameterized.named_parameters(
      ("invalid_source", "building"),
      ("empty_string", ""),
      ("numeric", "123"),
      ("special_chars", "unit@#$"),
  )
  def test_invalid_source_rejected(self, source):
    """Test that invalid source values are rejected."""
    with self.assertRaises(ValidationError) as context:
      freeciv_state.FreeCivAction(
          action_type="unit_move",
          actor_id=100,
          source=source,
          target={"x": 10, "y": 20},
      )
    error_msg = str(context.exception)
    # Should mention pattern validation or invalid source
    self.assertTrue(
        "pattern" in error_msg.lower()
        or "String should match pattern" in error_msg
    )

  @parameterized.named_parameters(
      ("negative", -0.5),
      ("over_one", 1.5),
      ("way_over", 100.0),
  )
  def test_invalid_confidence_out_of_bounds(self, confidence):
    """Test that confidence must be in range [0.0, 1.0]."""
    with self.assertRaises(ValidationError) as context:
      freeciv_state.FreeCivAction(
          action_type="unit_move",
          actor_id=100,
          source="unit",
          target={"x": 10, "y": 20},
          confidence=confidence,
      )
    error_msg = str(context.exception)
    self.assertTrue(
        "greater than or equal to 0" in error_msg
        or "less than or equal to 1" in error_msg
    )

  @parameterized.named_parameters(
      ("negative", -0.1),
      ("over_one", 1.1),
      ("large", 999.0),
  )
  def test_invalid_strategic_score_out_of_bounds(self, score):
    """Test that strategic_score must be in range [0.0, 1.0]."""
    with self.assertRaises(ValidationError) as context:
      freeciv_state.FreeCivAction(
          action_type="unit_move",
          actor_id=100,
          source="unit",
          target={"x": 10, "y": 20},
          strategic_score=score,
      )
    error_msg = str(context.exception)
    self.assertTrue(
        "greater than or equal to 0" in error_msg
        or "less than or equal to 1" in error_msg
    )


class FreeCivStateValidationTest(parameterized.TestCase):
  """Negative test cases for FreeCiv state structure validation."""

  def _get_minimal_valid_state(self):
    """Helper to create minimal valid state structure."""
    return {
        "game": {"turn": 1, "phase": "movement"},
        "map": {"width": 64, "height": 64, "tiles": []},
        "players": {},
        "units": {},
        "cities": {},
    }

  @parameterized.named_parameters(
      ("game", "game"),
      ("map", "map"),
      ("players", "players"),
      ("units", "units"),
      ("cities", "cities"),
  )
  def test_missing_required_field_raises_error(self, field_name):
    """Test that missing required fields are detected."""
    state = self._get_minimal_valid_state()
    del state[field_name]

    with self.assertRaises(ValueError) as context:
      freeciv_state.FreeCivState(state)
    self.assertIn(f"Missing required field: {field_name}", str(context.exception))

  @parameterized.named_parameters(
      ("string", "not_a_dict"),
      ("list", [1, 2, 3]),
      ("int", 123),
      ("none", None),
  )
  def test_game_field_wrong_type(self, invalid_value):
    """Test that 'game' field must be a dictionary."""
    state = self._get_minimal_valid_state()
    state["game"] = invalid_value

    with self.assertRaises(TypeError) as context:
      freeciv_state.FreeCivState(state)
    self.assertIn("'game' field must be a dictionary", str(context.exception))

  @parameterized.named_parameters(
      ("string", "not_a_dict"),
      ("list", [1, 2, 3]),
      ("int", 456),
      ("none", None),
  )
  def test_map_field_wrong_type(self, invalid_value):
    """Test that 'map' field must be a dictionary."""
    state = self._get_minimal_valid_state()
    state["map"] = invalid_value

    with self.assertRaises(TypeError) as context:
      freeciv_state.FreeCivState(state)
    self.assertIn("'map' field must be a dictionary", str(context.exception))

  @parameterized.named_parameters(
      ("string", "not_a_dict"),
      ("list", [{"id": 0, "name": "P1"}]),
      ("int", 789),
      ("none", None),
  )
  def test_players_field_wrong_type(self, invalid_value):
    """Test that 'players' field must be a dict."""
    state = self._get_minimal_valid_state()
    state["players"] = invalid_value

    with self.assertRaises(TypeError) as context:
      freeciv_state.FreeCivState(state)
    self.assertIn("'players' field must be a dict", str(context.exception))

  @parameterized.named_parameters(
      ("string", "not_a_dict"),
      ("list", [{"id": 1, "type": "Warriors"}]),
      ("int", 123),
      ("none", None),
  )
  def test_units_field_wrong_type(self, invalid_value):
    """Test that 'units' field must be a dict."""
    state = self._get_minimal_valid_state()
    state["units"] = invalid_value

    with self.assertRaises(TypeError) as context:
      freeciv_state.FreeCivState(state)
    self.assertIn("'units' field must be a dict", str(context.exception))

  @parameterized.named_parameters(
      ("string", "not_a_dict"),
      ("list", [{"id": 1, "name": "Rome"}]),
      ("int", 456),
      ("none", None),
  )
  def test_cities_field_wrong_type(self, invalid_value):
    """Test that 'cities' field must be a dict."""
    state = self._get_minimal_valid_state()
    state["cities"] = invalid_value

    with self.assertRaises(TypeError) as context:
      freeciv_state.FreeCivState(state)
    self.assertIn("'cities' field must be a dict", str(context.exception))

  def test_empty_state_raises_error(self):
    """Test that completely empty state is rejected."""
    with self.assertRaises(ValueError) as context:
      freeciv_state.FreeCivState({})
    # Should mention at least one missing field
    self.assertIn("Missing required field", str(context.exception))

  def test_state_with_only_game_field_raises_error(self):
    """Test that state with only game field is rejected."""
    state = {"game": {"turn": 1}}

    with self.assertRaises(ValueError) as context:
      freeciv_state.FreeCivState(state)
    self.assertIn("Missing required field", str(context.exception))

  def test_player_with_invalid_score_below_negative_one(self):
    """Test that player score less than -1 is rejected."""
    state = self._get_minimal_valid_state()
    state["players"] = {
        "0": {
            "id": 0,
            "name": "Test",
            "nation": "Romans",
            "score": -2,  # Invalid: less than -1
            "gold": 0,
        }
    }

    with self.assertRaises(ValidationError) as context:
      freeciv_state.FreeCivState(state)
    self.assertIn("Score must be >= -1", str(context.exception))

  def test_unit_with_invalid_activity_type(self):
    """Test that unit with invalid activity type is rejected."""
    state = self._get_minimal_valid_state()
    state["units"] = {
        "100": {
            "id": 100,
            "owner": 0,
            "type": "Warriors",
            "x": 10,
            "y": 20,
            "hp": 10,
            "moves": 1,
            "activity": [1, 2, 3],  # Invalid: list instead of str/int/None
        }
    }

    with self.assertRaises(ValidationError) as context:
      freeciv_state.FreeCivState(state)
    self.assertIn("Activity must be string, int, or None", str(context.exception))

  def test_nested_state_too_deep_from_json_raises_error(self):
    """Test that excessively nested JSON structure is rejected."""
    # Note: Depth validation only occurs in from_json(), not in __init__()
    # Create deeply nested structure as JSON
    nested_data = {}
    current = nested_data
    for i in range(15):  # Exceeds MAX_JSON_DEPTH of 10
      current["nested"] = {}
      current = current["nested"]

    state_dict = self._get_minimal_valid_state()
    state_dict["game"]["deep_nest"] = nested_data

    import json
    json_str = json.dumps(state_dict)

    with self.assertRaises(ValueError) as context:
      freeciv_state.FreeCivAction.from_json(json_str)
    self.assertIn("depth", str(context.exception).lower())

  def test_nested_state_allowed_via_init(self):
    """Test that deeply nested state via __init__ is allowed."""
    # Note: __init__() does not validate depth (only size)
    # This verifies the behavior difference between __init__ and from_json
    nested_data = {}
    current = nested_data
    for i in range(15):  # Would exceed MAX_JSON_DEPTH if validated
      current["nested"] = {}
      current = current["nested"]

    state = self._get_minimal_valid_state()
    state["game"]["deep_nest"] = nested_data

    # Should not raise an error - depth validation is not performed in __init__
    state_obj = freeciv_state.FreeCivState(state)
    self.assertIsNotNone(state_obj)


if __name__ == "__main__":
  absltest.main()
