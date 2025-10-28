"""Action conversion utilities for FreeCiv LLM Agent.

This module provides bidirectional conversion between FreeCivAction objects
and the integer action representations expected by the tournament framework.
It also handles conversion to/from string representations for LLM parsing.

Example usage:
    >>> from game_arena.harness.freeciv_action_converter import FreeCivActionConverter
    >>> from game_arena.harness.freeciv_state import FreeCivAction, FreeCivState
    >>>
    >>> converter = FreeCivActionConverter()
    >>> action = FreeCivAction("unit_move", 101, {"x": 2, "y": 3}, {}, "unit")
    >>> action_int = converter.action_to_int(action, state)
    >>> print(action_int)  # 42
    >>>
    >>> converted_back = converter.int_to_action(action_int, state)
    >>> print(converted_back.action_type)  # "unit_move"
"""

import hashlib
import json
from typing import Any, Dict, List, Optional

from absl import logging

from game_arena.harness.freeciv_cache import LRUCache
from game_arena.harness.freeciv_state import FreeCivAction, FreeCivState


class FreeCivActionConverter:
  """Converts between FreeCivAction objects and integer representations.

  This class provides bidirectional conversion between FreeCivAction objects
  (used internally by the agent) and integer action IDs (expected by the
  tournament framework). It maintains mappings and caching for performance.

  Attributes:
    action_cache: LRU cache for action conversions
    string_cache: LRU cache for string conversions
  """

  def __init__(self):
    """Initialize FreeCiv action converter."""
    # Cache for expensive conversion operations - configurable sizes and TTL
    from game_arena.harness.freeciv_proxy_client import (
        DEFAULT_ACTION_CACHE_SIZE, DEFAULT_STRING_CACHE_SIZE, DEFAULT_MEMORY_CACHE_TTL
    )
    self.action_cache = LRUCache[str, int](
        max_size=DEFAULT_ACTION_CACHE_SIZE, ttl_seconds=DEFAULT_MEMORY_CACHE_TTL
    )
    self.string_cache = LRUCache[str, str](
        max_size=DEFAULT_STRING_CACHE_SIZE, ttl_seconds=DEFAULT_MEMORY_CACHE_TTL
    )

    logging.debug("FreeCivActionConverter initialized")

  def action_to_int(self, action: FreeCivAction, state: FreeCivState, player_id: Optional[int] = None) -> int:
    """Convert FreeCivAction to action integer.

    Args:
      action: FreeCivAction object to convert
      state: Current game state for context
      player_id: Optional player ID (will be extracted if not provided)

    Returns:
      Integer representation of the action

    Raises:
      ValueError: If action cannot be converted or is invalid
    """
    # Validate inputs
    if not isinstance(action, FreeCivAction):
      raise ValueError(f"Expected FreeCivAction, got {type(action)}")

    # Determine and validate player ID
    if player_id is None:
      player_id = self._extract_player_id(action, state)

    # Validate player ID using security function
    from game_arena.harness.freeciv_proxy_client import validate_player_id
    player_id = validate_player_id(player_id)

    # Create cache key based on action and state
    cache_key = self._create_action_cache_key(action, state, player_id)

    # Check cache first
    cached_result = self.action_cache.get(cache_key)
    if cached_result is not None:
      logging.debug("Action conversion cache hit")
      return cached_result

    # Get legal actions from state
    legal_actions = state.get_legal_actions(player_id=player_id)

    # Find matching action in legal actions list
    action_index = self._find_action_index(action, legal_actions)

    if action_index == -1:
      raise ValueError(f"Action not found in legal actions: {action}")

    # Cache the result
    self.action_cache.set(cache_key, action_index)

    logging.debug(
        "Converted action to int: %s -> %d",
        self._action_to_string_key(action),
        action_index,
    )

    return action_index

  def int_to_action(
      self, action_int: int, state: FreeCivState, player_id: Optional[int] = None
  ) -> FreeCivAction:
    """Convert action integer to FreeCivAction.

    Args:
      action_int: Integer action ID
      state: Current game state for context
      player_id: Optional player ID (will default to current player if not provided)

    Returns:
      FreeCivAction object

    Raises:
      ValueError: If action_int is invalid or out of range
    """
    # Validate action integer input
    from game_arena.harness.freeciv_proxy_client import validate_action_id, validate_player_id
    action_int = validate_action_id(action_int)

    # Determine and validate player ID
    if player_id is None:
      player_id = self._extract_current_player_id(state)
    player_id = validate_player_id(player_id)

    legal_actions = state.get_legal_actions(player_id)

    if action_int < 0 or action_int >= len(legal_actions):
      raise ValueError(
          f"Action index {action_int} out of range [0, {len(legal_actions)-1}]"
      )

    action = legal_actions[action_int]

    logging.debug(
        "Converted int to action: %d -> %s",
        action_int,
        self._action_to_string_key(action),
    )

    return action

  def action_to_string(self, action: FreeCivAction) -> str:
    """Convert FreeCivAction to canonical string representation.

    Args:
      action: FreeCivAction to convert

    Returns:
      Canonical string representation of the action
    """
    cache_key = self._action_to_string_key(action)

    # Check cache
    cached_result = self.string_cache.get(cache_key)
    if cached_result is not None:
      return cached_result

    # Generate canonical string
    canonical_string = self._generate_canonical_string(action)

    # Cache result
    self.string_cache.set(cache_key, canonical_string)

    return canonical_string

  def string_to_action(self, action_string: str) -> FreeCivAction:
    """Convert canonical string to FreeCivAction.

    Args:
      action_string: Canonical string representation

    Returns:
      FreeCivAction object

    Raises:
      ValueError: If string format is invalid
    """
    # Parse the canonical string format
    # Example: "unit_move_warrior(101)_to(2,3)"
    try:
      return self._parse_canonical_string(action_string)
    except Exception as e:
      raise ValueError(f"Invalid action string format: {action_string}") from e

  def _find_action_index(
      self, target_action: FreeCivAction, legal_actions: List[FreeCivAction]
  ) -> int:
    """Find index of target action in legal actions list.

    Args:
      target_action: Action to find
      legal_actions: List of legal actions

    Returns:
      Index of action in list, or -1 if not found
    """
    for i, legal_action in enumerate(legal_actions):
      if self._actions_equal(target_action, legal_action):
        return i

    return -1

  def _actions_equal(
      self, action1: FreeCivAction, action2: FreeCivAction
  ) -> bool:
    """Check if two FreeCivActions are equivalent.

    Args:
      action1: First action
      action2: Second action

    Returns:
      True if actions are equivalent
    """
    return (
        action1.action_type == action2.action_type
        and action1.actor_id == action2.actor_id
        and action1.target == action2.target
        and action1.source == action2.source
    )

  def _extract_player_id(self, action: FreeCivAction, state: FreeCivState) -> int:
    """Extract player ID from action context and state.

    Args:
      action: FreeCivAction to analyze
      state: Current game state

    Returns:
      Player ID
    """
    # Try to extract from action parameters
    if hasattr(action, 'parameters') and action.parameters:
      if 'player_id' in action.parameters:
        return action.parameters['player_id']

    # Try to extract from action source/actor context
    if hasattr(action, 'actor_id') and action.actor_id:
      # Try to map actor_id to player_id
      player_id = self._actor_id_to_player_id(action.actor_id, state)
      if player_id is not None:
        return player_id

    # Fall back to current player from state
    return self._extract_current_player_id(state)

  def _extract_current_player_id(self, state: FreeCivState) -> int:
    """Extract current player ID from game state.

    Args:
      state: Current game state

    Returns:
      Current player ID
    """
    # Try to get current player from state method
    if hasattr(state, 'current_player') and callable(state.current_player):
      player_id = state.current_player()
      if player_id is not None:
        return player_id

    # Try to get from internal attribute
    if hasattr(state, '_current_player_id') and state._current_player_id > 0:
      return state._current_player_id

    # Default to player 1 if cannot determine
    logging.warning("Could not determine current player ID, defaulting to 1")
    return 1

  def _actor_id_to_player_id(self, actor_id: int, state: FreeCivState) -> Optional[int]:
    """Map actor ID to player ID using game state.

    Args:
      actor_id: Actor/unit ID
      state: Current game state

    Returns:
      Player ID if found, None otherwise
    """
    # Try to find the actor in units and get its owner
    if hasattr(state, 'units') and state.units:
      for unit in state.units:
        if isinstance(unit, dict):
          if unit.get('id') == actor_id:
            return unit.get('owner', unit.get('player_id'))
        elif hasattr(unit, 'id') and unit.id == actor_id:
          return getattr(unit, 'owner', getattr(unit, 'player_id', None))

    # Try to find the actor in cities
    if hasattr(state, 'cities') and state.cities:
      for city in state.cities:
        if isinstance(city, dict):
          if city.get('id') == actor_id:
            return city.get('owner', city.get('player_id'))
        elif hasattr(city, 'id') and city.id == actor_id:
          return getattr(city, 'owner', getattr(city, 'player_id', None))

    return None

  def _create_action_cache_key(
      self, action: FreeCivAction, state: FreeCivState, player_id: int
  ) -> str:
    """Create cache key for action conversion.

    Args:
      action: FreeCivAction object
      state: Current game state
      player_id: Player ID for the action

    Returns:
      Cache key string
    """
    action_data = {
        "action_type": action.action_type,
        "actor_id": action.actor_id,
        "target": action.target,
        "source": action.source,
        "turn": state.turn,
        "player_id": player_id,
    }

    action_str = json.dumps(action_data, sort_keys=True)
    return hashlib.sha256(action_str.encode()).hexdigest()[:16]

  def _action_to_string_key(self, action: FreeCivAction) -> str:
    """Create string key for action identification.

    Args:
      action: FreeCivAction object

    Returns:
      String key for action
    """
    return f"{action.action_type}_{action.actor_id}_{action.source}"

  def _generate_canonical_string(self, action: FreeCivAction) -> str:
    """Generate canonical string representation of action.

    Args:
      action: FreeCivAction to convert

    Returns:
      Canonical string format
    """
    if action.action_type == "unit_move" and action.target:
      return f"unit_move_{action.source}({action.actor_id})_to({action.target['x']},{action.target['y']})"

    elif action.action_type == "unit_attack" and action.target:
      return f"unit_attack_{action.source}({action.actor_id})_target({action.target['id']})"

    elif action.action_type in ["unit_fortify", "unit_explore"]:
      return f"{action.action_type}_{action.source}({action.actor_id})"

    elif action.action_type == "city_production" and action.target:
      target_value = (
          action.target.get("value")
          or action.target.get("name")
          or action.target.get("id")
      )
      return f"city_production_{action.source}({action.actor_id})_target({target_value})"

    elif action.action_type == "city_build_improvement" and action.target:
      target_value = (
          action.target.get("value")
          or action.target.get("name")
          or action.target.get("id")
      )
      return f"city_build_improvement_{action.source}({action.actor_id})_target({target_value})"

    elif action.action_type == "tech_research" and action.target:
      target_value = (
          action.target.get("value")
          or action.target.get("name")
          or action.target.get("tech")
      )
      return f"tech_research_player({action.actor_id})_target({target_value})"

    else:
      # Generic format
      target_str = ""
      if action.target:
        if "x" in action.target and "y" in action.target:
          target_str = f"_to({action.target['x']},{action.target['y']})"
        elif "id" in action.target:
          target_str = f"_target({action.target['id']})"
        elif "value" in action.target:
          target_str = f"_target({action.target['value']})"

      return (
          f"{action.action_type}_{action.source}({action.actor_id}){target_str}"
      )

  def _parse_canonical_string(self, action_string: str) -> FreeCivAction:
    """Parse canonical string to FreeCivAction.

    Args:
      action_string: Canonical string representation

    Returns:
      FreeCivAction object

    Raises:
      ValueError: If string format is invalid
    """
    import re

    # Parse unit_move format: unit_move_warrior(101)_to(2,3)
    move_match = re.match(
        r"unit_move_([^(]+)\((\d+)\)_to\((\d+),(\d+)\)", action_string
    )
    if move_match:
      unit_type, unit_id, x, y = move_match.groups()
      return FreeCivAction(
          action_type="unit_move",
          actor_id=int(unit_id),
          target={"x": int(x), "y": int(y)},
          parameters={},
          source="unit",
      )

    # Parse unit_attack format: unit_attack_warrior(101)_target(202)
    attack_match = re.match(
        r"unit_attack_([^(]+)\((\d+)\)_target\((\d+)\)", action_string
    )
    if attack_match:
      unit_type, unit_id, target_id = attack_match.groups()
      return FreeCivAction(
          action_type="unit_attack",
          actor_id=int(unit_id),
          target={"id": int(target_id)},
          parameters={},
          source="unit",
      )

    # Parse unit_fortify format: unit_fortify_warrior(101)
    fortify_match = re.match(r"unit_fortify_([^(]+)\((\d+)\)", action_string)
    if fortify_match:
      unit_type, unit_id = fortify_match.groups()
      return FreeCivAction(
          action_type="unit_fortify",
          actor_id=int(unit_id),
          target={},
          parameters={},
          source="unit",
      )

    # Parse city_production format: city_production_rome(301)_target(warriors)
    production_match = re.match(
        r"city_production_([^(]+)\((\d+)\)_target\(([^)]+)\)", action_string
    )
    if production_match:
      city_name, city_id, target_value = production_match.groups()
      return FreeCivAction(
          action_type="city_production",
          actor_id=int(city_id),
          target={"value": target_value},
          parameters={},
          source="city",
      )

    # Parse unit_explore format: unit_explore_warrior(101)
    explore_match = re.match(r"unit_explore_([^(]+)\((\d+)\)", action_string)
    if explore_match:
      unit_type, unit_id = explore_match.groups()
      return FreeCivAction(
          action_type="unit_explore",
          actor_id=int(unit_id),
          target={},
          parameters={},
          source="unit",
      )

    # Parse city_build_improvement format: city_build_improvement_rome(301)_target(barracks)
    improvement_match = re.match(
        r"city_build_improvement_([^(]+)\((\d+)\)_target\(([^)]+)\)", action_string
    )
    if improvement_match:
      city_name, city_id, improvement = improvement_match.groups()
      return FreeCivAction(
          action_type="city_build_improvement",
          actor_id=int(city_id),
          target={"value": improvement},
          parameters={},
          source="city",
      )

    # Parse tech_research format: tech_research_player(1)_target(Alphabet)
    tech_match = re.match(
        r"tech_research_player\((\d+)\)_target\(([^)]+)\)", action_string
    )
    if tech_match:
      player_id, tech_name = tech_match.groups()
      return FreeCivAction(
          action_type="tech_research",
          actor_id=int(player_id),
          target={"value": tech_name},
          parameters={},
          source="player",
      )

    # Generic parsing fallback
    # Format: action_type_source(actor_id)_target(value)
    generic_match = re.match(
        r"([^_]+)_([^(]+)\((\d+)\)(?:_target\(([^)]+)\))?", action_string
    )
    if generic_match:
      action_type, source, actor_id, target_value = generic_match.groups()
      target = {"value": target_value} if target_value else {}

      return FreeCivAction(
          action_type=action_type,
          actor_id=int(actor_id),
          target=target,
          parameters={},
          source=source,
      )

    raise ValueError(f"Cannot parse action string: {action_string}")

  def get_cache_statistics(self) -> Dict[str, Any]:
    """Get cache performance statistics.

    Returns:
      Dictionary with cache metrics
    """
    return {
        "action_cache": self.action_cache.statistics,
        "string_cache": self.string_cache.statistics,
    }

  def clear_caches(self) -> None:
    """Clear all caches."""
    self.action_cache.clear()
    self.string_cache.clear()
    logging.debug("Action converter caches cleared")
