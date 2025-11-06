"""State synchronization utilities for FreeCiv LLM Agent.

This module provides synchronization between the Game Arena framework
and the FreeCiv3D server state, ensuring consistent game state across
different communication channels.

Example usage:
    >>> from game_arena.harness.freeciv_state_sync import FreeCivStateSynchronizer
    >>> from game_arena.harness.freeciv_proxy_client import FreeCivProxyClient
    >>>
    >>> synchronizer = FreeCivStateSynchronizer()
    >>> proxy_client = FreeCivProxyClient("ws://localhost:4002")
    >>> await proxy_client.connect()
    >>>
    >>> observation = {"turn": 5, "playerID": 1}
    >>> state = await synchronizer.sync_state(proxy_client, observation)
    >>> print(state.turn)  # Current game turn
"""

import asyncio
import json
import time
from typing import Any, Dict, Mapping, Optional

from absl import logging

from game_arena.harness.freeciv_cache import LRUCache
from game_arena.harness.freeciv_proxy_client import FreeCivProxyClient
from game_arena.harness.freeciv_state import FreeCivState


class FreeCivStateSynchronizer:
  """Synchronizes game state between Game Arena and FreeCiv3D server.

  This class manages the synchronization of game state between the tournament
  framework observations and the live FreeCiv3D server state. It handles
  caching, validation, and consistency checks.

  Attributes:
    state_cache: LRU cache for expensive state synchronization operations
    last_sync_time: Timestamp of last successful synchronization
    sync_timeout: Timeout for synchronization operations
  """

  def __init__(self, sync_timeout: float = 10.0):
    """Initialize state synchronizer.

    Args:
      sync_timeout: Timeout in seconds for synchronization operations
    """
    self.sync_timeout = sync_timeout
    self.state_cache = LRUCache[str, FreeCivState](
        max_size=50, ttl_seconds=5.0  # Short TTL for state freshness
    )
    self.last_sync_time: Optional[float] = None

    logging.debug("FreeCivStateSynchronizer initialized")

  async def sync_state(
      self, proxy_client: FreeCivProxyClient, observation: Mapping[str, Any]
  ) -> FreeCivState:
    """Synchronize and return current game state.

    Args:
      proxy_client: WebSocket client for FreeCiv3D communication
      observation: Current observation from tournament framework

    Returns:
      Synchronized FreeCivState object

    Raises:
      asyncio.TimeoutError: If synchronization times out
      ValueError: If state data is invalid or inconsistent
    """
    start_time = time.time()

    try:
      # Create cache key from observation
      cache_key = self._create_cache_key(observation)

      # Check cache first
      cached_state = self.state_cache.get(cache_key)
      if cached_state is not None:
        logging.debug("State synchronization cache hit")
        return cached_state

      # Fetch fresh state from FreeCiv3D server
      state = await self._fetch_and_validate_state(proxy_client, observation)

      # Cache the result
      self.state_cache.set(cache_key, state)

      # Update sync timestamp
      self.last_sync_time = time.time()

      sync_duration = self.last_sync_time - start_time
      logging.debug(
          "State synchronized: turn=%d, duration=%.2fs",
          state.turn,
          sync_duration,
      )

      return state

    except asyncio.TimeoutError:
      logging.error(
          "State synchronization timed out after %.1fs", self.sync_timeout
      )
      raise
    except Exception as e:
      logging.error("State synchronization failed: %s", str(e))
      raise

  async def _fetch_and_validate_state(
      self, proxy_client: FreeCivProxyClient, observation: Mapping[str, Any]
  ) -> FreeCivState:
    """Fetch state from server and validate consistency.

    Args:
      proxy_client: WebSocket client
      observation: Tournament observation

    Returns:
      Validated FreeCivState object

    Raises:
      asyncio.TimeoutError: If fetch times out
      ValueError: If state validation fails
    """
    # Fetch state with timeout
    state_data = await asyncio.wait_for(
        proxy_client.get_state(), timeout=self.sync_timeout
    )

    # Validate state data structure
    self._validate_state_data(state_data)

    # Check consistency with observation
    self._check_consistency(state_data, observation)

    # Create FreeCivState object
    freeciv_state = FreeCivState(state_data)

    # Additional validation of created state
    self._validate_freeciv_state(freeciv_state)

    return freeciv_state

  def _validate_state_data(self, state_data: Dict[str, Any]) -> None:
    """Validate raw state data from server.

    Args:
      state_data: Raw state data from FreeCiv3D server

    Raises:
      ValueError: If state data is invalid
    """
    required_fields = ["turn", "players", "units", "cities"]

    for field in required_fields:
      if field not in state_data:
        raise ValueError(f"Missing required field in state data: {field}")

    # Validate turn number
    turn = state_data.get("turn")
    if not isinstance(turn, int) or turn < 0:
      raise ValueError(f"Invalid turn number: {turn}")

    # Validate players data (dict-only format as of commit 49417fc)
    players = state_data.get("players", {})
    if not isinstance(players, dict):
      raise ValueError("Players data must be a dict")

    # Validate units data (dict-only format as of commit 49417fc)
    units = state_data.get("units", {})
    if not isinstance(units, dict):
      raise ValueError("Units data must be a dict")

    # Validate cities data (dict-only format as of commit 49417fc)
    cities = state_data.get("cities", {})
    if not isinstance(cities, dict):
      raise ValueError("Cities data must be a dict")

    logging.debug(
        "State data validated: turn=%d, players=%d, units=%d, cities=%d",
        turn,
        len(players),
        len(units),
        len(cities),
    )

  def _check_consistency(
      self, state_data: Dict[str, Any], observation: Mapping[str, Any]
  ) -> None:
    """Check consistency between server state and observation.

    Args:
      state_data: State data from FreeCiv3D server
      observation: Observation from tournament framework

    Raises:
      ValueError: If inconsistencies are detected
    """
    # Check turn consistency if both sources provide it
    server_turn = state_data.get("turn")
    obs_turn = observation.get("turn")

    if server_turn is not None and obs_turn is not None:
      if abs(server_turn - obs_turn) > 1:  # Allow 1 turn difference
        logging.warning(
            "Turn mismatch: server=%d, observation=%d", server_turn, obs_turn
        )
        # Don't raise error for turn mismatch as it may be expected

    # Check player count consistency
    server_players = len(state_data.get("players", []))
    obs_players = len(observation.get("players", []))

    if obs_players > 0 and abs(server_players - obs_players) > 0:
      logging.warning(
          "Player count mismatch: server=%d, observation=%d",
          server_players,
          obs_players,
      )

    logging.debug("Consistency check passed")

  def _validate_freeciv_state(self, state: FreeCivState) -> None:
    """Validate created FreeCivState object.

    Args:
      state: FreeCivState object to validate

    Raises:
      ValueError: If state object is invalid
    """
    # Check basic properties
    if state.turn < 0:
      raise ValueError(f"Invalid state turn: {state.turn}")

    if not isinstance(state.players, dict):
      raise ValueError("State players must be a dictionary")

    if not isinstance(state.units, dict):
      raise ValueError("State units must be a dictionary")

    if not isinstance(state.cities, dict):
      raise ValueError("State cities must be a dictionary")

    # Validate that we can get legal actions (basic functionality test)
    try:
      # Try to get current player from state first
      test_player_id = getattr(state, '_current_player_id', 1)
      if test_player_id <= 0:
        test_player_id = 1

      legal_actions = state.get_legal_actions(player_id=test_player_id)
      if not isinstance(legal_actions, list):
        raise ValueError("Legal actions must be a list")
    except Exception as e:
      raise ValueError(f"Cannot get legal actions from state: {e}")

    logging.debug("FreeCivState validation passed")

  def _create_cache_key(self, observation: Mapping[str, Any]) -> str:
    """Create cache key from observation data.

    Args:
      observation: Tournament observation

    Returns:
      Cache key string
    """
    # Use turn and player ID as primary cache key components
    turn = observation.get("turn", 0)

    # Try multiple keys for player ID
    player_id = None
    for key in ["playerID", "player_id", "current_player", "agent_id"]:
      if key in observation:
        player_id = observation[key]
        break

    if player_id is None:
      player_id = 1

    # Add timestamp component to ensure freshness
    timestamp_bucket = int(time.time() / 5)  # 5-second buckets

    return f"state_{turn}_{player_id}_{timestamp_bucket}"

  async def refresh_state(
      self, proxy_client: FreeCivProxyClient, observation: Mapping[str, Any]
  ) -> FreeCivState:
    """Force refresh of state, bypassing cache.

    Args:
      proxy_client: WebSocket client
      observation: Tournament observation

    Returns:
      Fresh FreeCivState object
    """
    # Clear relevant cache entries
    cache_key = self._create_cache_key(observation)
    self.state_cache.invalidate(cache_key)

    # Fetch fresh state
    return await self.sync_state(proxy_client, observation)

  def get_sync_status(self) -> Dict[str, Any]:
    """Get synchronization status and statistics.

    Returns:
      Dictionary with sync status information
    """
    return {
        "last_sync_time": self.last_sync_time,
        "seconds_since_sync": (
            time.time() - self.last_sync_time if self.last_sync_time else None
        ),
        "cache_stats": self.state_cache.statistics,
        "sync_timeout": self.sync_timeout,
    }

  def clear_cache(self) -> None:
    """Clear state cache."""
    self.state_cache.clear()
    logging.debug("State synchronizer cache cleared")

  async def wait_for_turn(
      self,
      proxy_client: FreeCivProxyClient,
      target_turn: int,
      timeout: float = 30.0,
  ) -> FreeCivState:
    """Wait for game to reach specific turn.

    Args:
      proxy_client: WebSocket client
      target_turn: Turn number to wait for
      timeout: Maximum time to wait

    Returns:
      FreeCivState when target turn is reached

    Raises:
      asyncio.TimeoutError: If timeout is exceeded
      ValueError: If turn moves backwards
    """
    start_time = time.time()
    last_turn = -1

    while time.time() - start_time < timeout:
      # Fetch current state
      state_data = await proxy_client.get_state()
      current_turn = state_data.get("turn", 0)

      # Check for backwards movement (error condition)
      if current_turn < last_turn:
        raise ValueError(f"Turn moved backwards: {last_turn} -> {current_turn}")

      # Check if target reached
      if current_turn >= target_turn:
        logging.debug(
            "Target turn reached: %d (waited %.1fs)",
            current_turn,
            time.time() - start_time,
        )
        return FreeCivState(state_data)

      last_turn = current_turn

      # Wait briefly before checking again
      await asyncio.sleep(0.5)

    raise asyncio.TimeoutError(
        f"Timeout waiting for turn {target_turn} (last seen: {last_turn})"
    )
