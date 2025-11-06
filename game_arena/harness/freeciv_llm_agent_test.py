"""Unit tests for FreeCiv LLM Agent turn management functionality.

Tests the heuristic end_turn logic including:
- Turn state tracking
- Action history per turn
- Intelligent end_turn decision making
"""

import unittest
from unittest.mock import Mock

from game_arena.harness.freeciv_llm_agent import FreeCivLLMAgent
from game_arena.harness.freeciv_state import FreeCivAction, FreeCivState, FreeCivUnit


class TurnManagementTest(unittest.TestCase):
  """Tests for turn state tracking and end_turn heuristics."""

  def setUp(self):
    """Set up test fixtures."""
    # Create mock model
    self.mock_model = Mock()
    self.mock_model.model_name = "test-model"

    # Create agent with turn tracking
    self.agent = FreeCivLLMAgent(
        model=self.mock_model,
        strategy="balanced",
        use_rethinking=False
    )

  def test_initial_turn_state(self):
    """Test that agent initializes with correct turn state."""
    self.assertEqual(self.agent.current_turn, 1)
    self.assertEqual(len(self.agent.actions_this_turn), 0)
    self.assertEqual(self.agent.queries_without_action, 0)
    self.assertFalse(self.agent.last_turn_units_exhausted)

  def test_on_state_update_turn_change(self):
    """Test that on_state_update detects turn changes and resets state."""
    # Take some actions in turn 1
    action1 = FreeCivAction(
        action_type="unit_move",
        actor_id=132,
        target={"x": 10, "y": 20},
        source="unit"
    )
    self.agent.on_action_taken(action1)

    # Manually increment queries (simulating state queries without actions)
    self.agent.queries_without_action = 3

    # Verify state before turn change
    self.assertEqual(len(self.agent.actions_this_turn), 1)
    self.assertEqual(self.agent.queries_without_action, 3)  # Manually set above

    # Simulate turn advance
    game_state = {"turn": 2, "phase": "movement", "players": [], "units": [], "cities": []}
    self.agent.on_state_update(game_state)

    # Verify state was reset
    self.assertEqual(self.agent.current_turn, 2)
    self.assertEqual(len(self.agent.actions_this_turn), 0)
    self.assertEqual(self.agent.queries_without_action, 0)
    self.assertFalse(self.agent.last_turn_units_exhausted)

  def test_on_state_update_no_change(self):
    """Test that on_state_update preserves state when turn doesn't change."""
    # Take action in turn 1
    action1 = FreeCivAction(
        action_type="unit_move",
        actor_id=132,
        target={"x": 10, "y": 20},
        source="unit"
    )
    self.agent.on_action_taken(action1)

    # Same turn number
    game_state = {"turn": 1, "phase": "movement", "players": [], "units": [], "cities": []}
    self.agent.on_state_update(game_state)

    # State should be preserved
    self.assertEqual(self.agent.current_turn, 1)
    self.assertEqual(len(self.agent.actions_this_turn), 1)

  def test_on_action_taken_resets_query_counter(self):
    """Test that on_action_taken resets queries_without_action."""
    self.agent.queries_without_action = 5

    action = FreeCivAction(
        action_type="unit_move",
        actor_id=132,
        target={"x": 10, "y": 20},
        source="unit"
    )
    self.agent.on_action_taken(action)

    self.assertEqual(self.agent.queries_without_action, 0)

  def test_on_action_taken_excludes_end_turn(self):
    """Test that on_action_taken doesn't add end_turn to history."""
    end_turn = FreeCivAction(
        action_type="end_turn",
        actor_id=1,
        target=None,
        source="player"
    )
    self.agent.on_action_taken(end_turn)

    # end_turn should NOT be in actions_this_turn
    self.assertEqual(len(self.agent.actions_this_turn), 0)

  def test_should_end_turn_action_limit(self):
    """Test that should_end_turn returns True when approaching action limit."""
    # Create minimal state
    state = self._create_minimal_state()

    # Add 19 actions (max is 20, so this should trigger end_turn)
    for i in range(19):
      action = FreeCivAction(
          action_type="unit_move",
          actor_id=132 + i,
          target={"x": 10, "y": 20},
          source="unit"
      )
      self.agent.on_action_taken(action)

    result = self.agent.should_end_turn(state, player_id=0, max_actions_per_turn=20)
    self.assertTrue(result)

  def test_should_end_turn_all_units_exhausted(self):
    """Test that should_end_turn returns True when all units exhausted."""
    # Create state with units that have no moves left
    raw_state = {
        "turn": 1,
        "phase": "movement",
        "game": {"turn": 1, "phase": "movement"},
        "map": {"width": 64, "height": 64, "tiles": []},
        "players": [{"id": 0, "name": "P1", "nation": "Romans", "score": 100, "gold": 50}],
        "units": [
            {"id": 132, "owner": 0, "type": "Settlers", "x": 10, "y": 20, "hp": 10, "moves": 0, "moves_left": 0},
            {"id": 133, "owner": 0, "type": "Warriors", "x": 15, "y": 25, "hp": 10, "moves": 0, "moves_left": 0},
        ],
        "cities": [],
        "legal_actions": [],
    }
    state = FreeCivState(raw_state)

    # Take at least 1 action
    action = FreeCivAction(
        action_type="unit_move",
        actor_id=132,
        target={"x": 11, "y": 20},
        source="unit"
    )
    self.agent.on_action_taken(action)

    result = self.agent.should_end_turn(state, player_id=0, max_actions_per_turn=20)
    self.assertTrue(result)
    self.assertTrue(self.agent.last_turn_units_exhausted)

  def test_should_end_turn_stuck_without_progress(self):
    """Test that should_end_turn returns True when stuck (5+ queries without action)."""
    state = self._create_minimal_state()

    # Simulate 5 queries without taking action
    self.agent.queries_without_action = 5

    result = self.agent.should_end_turn(state, player_id=0, max_actions_per_turn=20)
    self.assertTrue(result)

  def test_should_end_turn_normal_gameplay(self):
    """Test that should_end_turn returns False during normal gameplay."""
    # Create state with units that have moves left
    raw_state = {
        "turn": 1,
        "phase": "movement",
        "game": {"turn": 1, "phase": "movement"},
        "map": {"width": 64, "height": 64, "tiles": []},
        "players": [{"id": 0, "name": "P1", "nation": "Romans", "score": 100, "gold": 50}],
        "units": [
            {"id": 132, "owner": 0, "type": "Settlers", "x": 10, "y": 20, "hp": 10, "moves": 3, "moves_left": 3},
            {"id": 133, "owner": 0, "type": "Warriors", "x": 15, "y": 25, "hp": 10, "moves": 1, "moves_left": 1},
        ],
        "cities": [],
        "legal_actions": [],
    }
    state = FreeCivState(raw_state)

    # Take 3 actions (below limit, units have moves, not stuck)
    for i in range(3):
      action = FreeCivAction(
          action_type="unit_move",
          actor_id=132 + i,
          target={"x": 10, "y": 20},
          source="unit"
      )
      self.agent.on_action_taken(action)

    self.agent.queries_without_action = 1

    result = self.agent.should_end_turn(state, player_id=0, max_actions_per_turn=20)
    self.assertFalse(result)

  def test_should_end_turn_no_units(self):
    """Test behavior when player has no units."""
    raw_state = {
        "turn": 1,
        "phase": "movement",
        "game": {"turn": 1, "phase": "movement"},
        "map": {"width": 64, "height": 64, "tiles": []},
        "players": [{"id": 0, "name": "P1", "nation": "Romans", "score": 100, "gold": 50}],
        "units": [],  # No units
        "cities": [],
        "legal_actions": [],
    }
    state = FreeCivState(raw_state)

    # Should not end turn just because no units (might be pre-game state)
    result = self.agent.should_end_turn(state, player_id=0, max_actions_per_turn=20)
    self.assertFalse(result)

  def _create_minimal_state(self):
    """Helper to create minimal FreeCivState for testing."""
    raw_state = {
        "turn": 1,
        "phase": "movement",
        "game": {"turn": 1, "phase": "movement"},
        "map": {"width": 64, "height": 64, "tiles": []},
        "players": [{"id": 0, "name": "P1", "nation": "Romans", "score": 100, "gold": 50}],
        "units": [{"id": 132, "owner": 0, "type": "Settlers", "x": 10, "y": 20, "hp": 10, "moves": 3, "moves_left": 3}],
        "cities": [],
        "legal_actions": [],
    }
    return FreeCivState(raw_state)


if __name__ == "__main__":
  unittest.main()
