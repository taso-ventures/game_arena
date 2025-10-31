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
        "players": [
            {
                "id": 0,
                "name": "Player1",
                "nation": "Romans",
                "score": 100,
                "gold": 50,
            }
        ],
        "units": [
            {
                "id": 132,
                "owner": 0,
                "type": "Settlers",
                "x": 10,
                "y": 20,
                "hp": 10,
                "moves": 3,
                "available_actions": [],  # Empty per-unit list
            }
        ],
        "cities": [],
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
        "players": [
            {"id": 0, "name": "P1", "nation": "Romans", "score": 100, "gold": 50},
            {"id": 1, "name": "P2", "nation": "Greeks", "score": 95, "gold": 45},
        ],
        "units": [
            {"id": 132, "owner": 0, "type": "Settlers", "x": 10, "y": 20, "hp": 10, "moves": 3},
            {"id": 133, "owner": 1, "type": "Warriors", "x": 15, "y": 25, "hp": 10, "moves": 1},
        ],
        "cities": [],
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
        "players": [{"id": 0, "name": "P1", "nation": "Romans", "score": 100, "gold": 50}],
        "units": [{"id": 132, "owner": 0, "type": "Settlers", "x": 10, "y": 20, "hp": 10, "moves": 3}],
        "cities": [],
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
        "players": [{"id": 0, "name": "P1", "nation": "Romans", "score": 100, "gold": 50}],
        "units": [{"id": 132, "owner": 0, "type": "Settlers", "x": 10, "y": 20, "hp": 10, "moves": 3}],
        "cities": [],
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
        "players": [{"id": 0, "name": "P1", "nation": "Romans", "score": 100, "gold": 50}],
        "units": [{"id": 132, "owner": 0, "type": "Settlers", "x": 10, "y": 20, "hp": 10, "moves": 3}],
        "cities": [],
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
        "players": [{"id": 0, "name": "P1", "nation": "Romans", "score": 100, "gold": 50}],
        "units": [{"id": 132, "owner": 0, "type": "Settlers", "x": 10, "y": 20, "hp": 10, "moves": 3}],
        "cities": [],
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
        "players": [{"id": 0, "name": "P1", "nation": "Romans", "score": 100, "gold": 50}],
        "units": [{"id": 132, "owner": 0, "type": "Settlers", "x": 10, "y": 20, "hp": 10, "moves": 3}],
        "cities": [],
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
        "players": [{"id": 0, "name": "P1", "nation": "Romans", "score": 100, "gold": 50}],
        "units": [{"id": 132, "owner": 0, "type": "Settlers", "x": 10, "y": 20, "hp": 10, "moves": 3}],
        "cities": [],
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


if __name__ == "__main__":
  absltest.main()
