"""FreeCiv3D LLM Gateway WebSocket Protocol Models.

This module provides Pydantic models for the FreeCiv3D LLM Gateway WebSocket
protocol as specified in freeciv3d/docs/llm_websocket_protocol.md.

The FreeCiv3D Gateway transforms state_response messages from the proxy by
flattening the nested state data to the top level, creating state_update
messages with the following structure:

    {
        "type": "state_update",
        "turn": 15,
        "phase": "movement",
        "game": {...},
        "map": {...},
        "players": {"0": {...}, "1": {...}},  # Dict keyed by player_id
        "units": {"101": {...}, "102": {...}},  # Dict keyed by unit_id
        "cities": {"201": {...}, "202": {...}},  # Dict keyed by city_id
        "format": "full",
        "cached": false,
        "timestamp": 1234567890.123
    }

This module provides:
  1. Pydantic models for protocol validation
  2. Utility functions to extract and transform data for FreeCivState

Example usage:
    >>> from game_arena.harness.freeciv_protocol_models import (
    ...     parse_state_update,
    ...     extract_game_state_for_freeciv_state,
    ... )
    >>>
    >>> # Validate and parse incoming message
    >>> state_msg = parse_state_update(websocket_message)
    >>>
    >>> # Extract data in FreeCivState-compatible format
    >>> game_state = extract_game_state_for_freeciv_state(state_msg)
    >>> # game_state has players/units/cities as lists
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Base Message Models
# =============================================================================


class FreeCiv3DMessage(BaseModel):
  """Base structure for all FreeCiv3D protocol messages.

  All messages from the FreeCiv3D LLM Gateway include a type field and
  optional timestamp.
  """

  type: str = Field(..., description="Message type identifier")
  timestamp: Optional[float] = Field(
      None, description="Unix timestamp of message creation"
  )


# =============================================================================
# Authentication Response Models
# =============================================================================


class AuthSuccessResponse(FreeCiv3DMessage):
  """AUTH_SUCCESS message from FreeCiv3D Gateway.

  Sent after successful authentication with game_id and password.
  """

  type: str = Field(default="auth_success", frozen=True)
  message: str = Field(..., description="Success message")
  game_id: str = Field(..., description="Authenticated game ID")


class ErrorResponse(FreeCiv3DMessage):
  """ERROR message from FreeCiv3D Gateway.

  Sent when an error occurs during message processing or game state access.
  """

  type: str = Field(default="error", frozen=True)
  error: str = Field(..., description="Error type or code")
  message: str = Field(..., description="Human-readable error message")
  details: Optional[Dict[str, Any]] = Field(
      None, description="Additional error context"
  )


# =============================================================================
# Game State Data Models
# =============================================================================


class GameDict(BaseModel):
  """Game metadata in state_update messages.

  Contains general game information like ruleset, difficulty, and victory
  conditions.
  """

  ruleset: Optional[str] = Field(None, description="Active ruleset name")
  difficulty: Optional[str] = Field(None, description="AI difficulty level")
  victory_conditions: Optional[List[str]] = Field(
      None, description="Active victory conditions"
  )
  turn_timeout: Optional[int] = Field(
      None, description="Turn timeout in seconds"
  )

  class Config:
    extra = "allow"  # Allow additional fields from server


class MapDict(BaseModel):
  """Map information in state_update messages.

  Contains map dimensions, topology, and terrain data.
  """

  width: Optional[int] = Field(None, description="Map width in tiles")
  height: Optional[int] = Field(None, description="Map height in tiles")
  topology: Optional[str] = Field(None, description="Map topology type")
  tiles: Optional[List[Dict[str, Any]]] = Field(
      None, description="Tile data array"
  )

  class Config:
    extra = "allow"


class PlayerDict(BaseModel):
  """Player data in state_update messages.

  Players are sent as a dict keyed by player_id strings:
      {"0": {...}, "1": {...}, "2": {...}}

  Each player entry contains game state for that player.
  """

  player_id: Optional[int] = Field(
      None, alias="playerno", description="Numeric player ID"
  )
  name: Optional[str] = Field(None, description="Player/nation name")
  nation: Optional[str] = Field(None, description="Nation identifier or ID")
  is_alive: Optional[bool] = Field(None, description="Player is active")
  is_ai: Optional[bool] = Field(None, description="AI-controlled player")
  gold: Optional[int] = Field(None, description="Treasury gold")
  science: Optional[int] = Field(None, description="Science points")
  government: Optional[str] = Field(None, description="Government type")
  researching: Optional[str] = Field(
      None, description="Current research target"
  )

  @field_validator("nation", mode="before")
  @classmethod
  def coerce_nation_to_string(cls, v: Any) -> Optional[str]:
    """Convert nation int ID to string.

    CivCom returns nation as int (nation ID from PACKET_PLAYER_INFO).
    We coerce to string for consistency.
    """
    if v is None:
      return None
    if isinstance(v, int):
      return str(v)
    return v

  class Config:
    extra = "allow"
    populate_by_name = True  # Allow both player_id and playerno


class UnitDict(BaseModel):
  """Unit data in state_update messages.

  Units are sent as a dict keyed by unit_id strings:
      {"101": {...}, "102": {...}, "103": {...}}

  Each unit entry contains unit state and capabilities.
  """

  unit_id: Optional[int] = Field(None, alias="id", description="Numeric unit ID")
  type: Optional[str] = Field(None, description="Unit type name")
  owner: Optional[int] = Field(None, description="Owning player ID")
  x: Optional[int] = Field(None, description="X coordinate")
  y: Optional[int] = Field(None, description="Y coordinate")
  moves_left: Optional[int] = Field(
      None, alias="movesleft", description="Remaining movement points"
  )
  hp: Optional[int] = Field(None, description="Current hit points")
  veteran_level: Optional[int] = Field(
      None, description="Veteran experience level"
  )
  activity: Optional[str] = Field(None, description="Current activity or idle state")

  @field_validator("activity", mode="before")
  @classmethod
  def coerce_activity(cls, v: Any) -> Optional[str]:
    """Convert activity int enum to string.

    CivCom returns activity as int (activity enum from PACKET_UNIT_INFO).
    Activity codes reference: FreeCiv server/unittools.h activity enum
    """
    # Mapping of FreeCiv activity codes to descriptive names
    ACTIVITY_MAPPING = {
        0: None,  # ACTIVITY_IDLE - no activity
        1: "pollution",  # ACTIVITY_POLLUTION
        2: "unused_road",  # deprecated
        3: "mine",  # ACTIVITY_MINE
        4: "irrigate",  # ACTIVITY_IRRIGATE
        5: "fortified",  # ACTIVITY_FORTIFIED
        6: "fortress",  # ACTIVITY_FORTRESS
        7: "sentry",  # ACTIVITY_SENTRY
        8: "unused_railroad",  # deprecated
        9: "pillage",  # ACTIVITY_PILLAGE
        10: "goto",  # ACTIVITY_GOTO
        11: "explore",  # ACTIVITY_EXPLORE
        12: "transform",  # ACTIVITY_TRANSFORM
        13: "unused_airbase",  # deprecated
        14: "fortifying",  # ACTIVITY_FORTIFYING
        15: "fallout",  # ACTIVITY_FALLOUT
        16: "unknown",  # ACTIVITY_UNKNOWN
        17: "patrol",  # ACTIVITY_PATROL
        18: "convert",  # ACTIVITY_CONVERT
        19: "cultivate",  # ACTIVITY_CULTIVATE
        20: "plant",  # ACTIVITY_PLANT
        21: "gen_road",  # ACTIVITY_GEN_ROAD
    }

    if v is None:
      return None
    if isinstance(v, int):
      return ACTIVITY_MAPPING.get(v, f"unknown_activity_{v}")
    return v

  class Config:
    extra = "allow"
    populate_by_name = True


class CityDict(BaseModel):
  """City data in state_update messages.

  Cities are sent as a dict keyed by city_id strings:
      {"201": {...}, "202": {...}, "203": {...}}

  Each city entry contains city state and production info.
  """

  city_id: Optional[int] = Field(None, alias="id", description="Numeric city ID")
  name: Optional[str] = Field(None, description="City name")
  owner: Optional[int] = Field(None, description="Owning player ID")
  x: Optional[int] = Field(None, description="X coordinate")
  y: Optional[int] = Field(None, description="Y coordinate")
  size: Optional[int] = Field(None, description="City population size")
  production: Optional[str] = Field(
      None, description="Current production target"
  )
  food_stock: Optional[int] = Field(None, description="Stored food")
  shield_stock: Optional[int] = Field(None, description="Stored shields")

  class Config:
    extra = "allow"
    populate_by_name = True


# =============================================================================
# State Update Response Model
# =============================================================================


class StateUpdateResponse(FreeCiv3DMessage):
  """STATE_UPDATE message from FreeCiv3D Gateway.

  This is the primary message type containing game state. The gateway flattens
  the state_response from the proxy, moving all state data to the top level.

  IMPORTANT: The FreeCiv3D team has chosen to keep the DICT format (Option A).
  Players, units, and cities are sent as DICTS keyed by their ID strings.
  This is more efficient and provides better API design.

  The extraction function in game_arena converts dicts to lists for FreeCivState.
  """

  type: str = Field(default="state_update", frozen=True)

  # Core game state
  turn: int = Field(..., description="Current game turn number")
  phase: Optional[str] = Field(
      "movement", description="Current game phase (movement, combat, etc.)"
  )
  player_id: Optional[int] = Field(
      None, description="Player ID this state is for (perspective)"
  )

  # Game metadata - OPTIONAL (missing in emergency fallback case)
  game: Optional[GameDict] = Field(
      None, description="Game metadata and settings"
  )
  map: Optional[MapDict] = Field(None, description="Map information and terrain")

  # Entity collections - server sends DICTS keyed by ID strings
  players: Dict[str, PlayerDict] = Field(
      default_factory=dict,
      description="Players dict keyed by player_id strings (e.g., {'0': {...}, '1': {...}})",
  )
  units: Dict[str, UnitDict] = Field(
      default_factory=dict,
      description="Units dict keyed by unit_id strings (e.g., {'101': {...}, '102': {...}})",
  )
  cities: Dict[str, CityDict] = Field(
      default_factory=dict,
      description="Cities dict keyed by city_id strings (e.g., {'301': {...}, '302': {...}})",
  )

  # Legal actions for current player (from FreeCiv3D LLM Gateway)
  legal_actions: Optional[List[Any]] = Field(
      None,
      description="Legal actions for current player (generated by FreeCiv3D)",
  )

  # LLM-optimized format fields (only present when format="llm_optimized")
  # These provide strategic analysis from FreeCiv3D to help LLMs make better decisions
  strategic_summary: Optional[Dict[str, Any]] = Field(
      None,
      description="High-level strategic situation (cities_count, units_count, tech_progress, military_strength)",
  )
  immediate_priorities: Optional[List[str]] = Field(
      None,
      description="Prioritized list of immediate actions to consider (e.g., 'explore_nearby_areas', 'build_military_units')",
  )
  threats: Optional[List[Dict[str, Any]]] = Field(
      None,
      description="Current threats to the player (military, economic, diplomatic)",
  )
  opportunities: Optional[List[Dict[str, Any]]] = Field(
      None,
      description="Strategic opportunities available (expansion sites, resource tiles, tech advantages)",
  )

  # Response metadata
  format: Optional[str] = Field("full", description="Response format type")
  cached: Optional[bool] = Field(False, description="Response was cached")


# =============================================================================
# Action Response Models
# =============================================================================


class ActionResultResponse(FreeCiv3DMessage):
  """ACTION_RESULT message from FreeCiv3D Gateway.

  Sent after executing an action to indicate success or failure.
  """

  type: str = Field(default="action_result", frozen=True)
  success: bool = Field(..., description="Action execution succeeded")
  action_type: str = Field(..., description="Type of action executed")
  message: Optional[str] = Field(None, description="Result message")
  error: Optional[str] = Field(None, description="Error message if failed")


# =============================================================================
# Other Protocol Messages
# =============================================================================


class WelcomeResponse(FreeCiv3DMessage):
  """WELCOME message from FreeCiv3D Gateway.

  Sent when a client first connects, before authentication.
  """

  type: str = Field(default="welcome", frozen=True)
  message: str = Field(..., description="Welcome message")
  version: Optional[str] = Field(None, description="Gateway version")


class GameReadyResponse(FreeCiv3DMessage):
  """GAME_READY message from FreeCiv3D Gateway.

  Sent when the game is initialized and ready for play.
  """

  type: str = Field(default="game_ready", frozen=True)
  game_id: str = Field(..., description="Game identifier")
  players: Optional[List[str]] = Field(
      None, description="Player names/identifiers"
  )


class LLMConnectResponse(FreeCiv3DMessage):
  """LLM_CONNECT message from FreeCiv3D Gateway.

  Sent to acknowledge LLM client connection.
  """

  type: str = Field(default="llm_connect", frozen=True)
  message: str = Field(..., description="Connection acknowledgment message")


# =============================================================================
# Utility Functions
# =============================================================================


def parse_state_update(message: Dict[str, Any]) -> StateUpdateResponse:
  """Parse and validate a state_update message.

  Args:
    message: Raw message dict from WebSocket

  Returns:
    Validated StateUpdateResponse object

  Raises:
    ValidationError: If message doesn't match protocol schema
    ValueError: If message type is not state_update
  """
  if message.get("type") != "state_update":
    raise ValueError(
        f"Expected state_update message, got type={message.get('type')}"
    )

  return StateUpdateResponse.model_validate(message)


def extract_game_state_for_freeciv_state(
    state_msg: StateUpdateResponse,
) -> Dict[str, Any]:
  """Extract game state in FreeCivState-compatible format.

  The FreeCiv3D proxy sends players, units, and cities as DICTS keyed by ID.
  FreeCivState also expects DICTS (as of commit 49417fc), so we preserve the
  dict structure while converting Pydantic models to plain dicts.

  This provides a clean separation:
  - Protocol layer: validates dict format from server
  - Extraction layer: converts Pydantic models to plain dicts (preserves structure)
  - Game state layer: consumes dicts keyed by ID

  Args:
    state_msg: Validated StateUpdateResponse from parse_state_update()

  Returns:
    Dict with structure:
      {
        "turn": int,
        "phase": str,
        "game": dict (or empty dict if None),
        "map": dict (or empty dict if None),
        "players": Dict[str, dict],  # Dict keyed by player ID
        "units": Dict[str, dict],    # Dict keyed by unit ID
        "cities": Dict[str, dict],   # Dict keyed by city ID
      }
  """
  # Keep dicts keyed by ID (dict-only format as of commit 49417fc)
  # Convert Pydantic models to plain dicts while preserving the dict structure
  players_dicts = {
      player_id: p.model_dump(by_alias=False)
      for player_id, p in state_msg.players.items()
  }
  units_dicts = {
      unit_id: u.model_dump(by_alias=False)
      for unit_id, u in state_msg.units.items()
  }
  cities_dicts = {
      city_id: c.model_dump(by_alias=False)
      for city_id, c in state_msg.cities.items()
  }

  # Handle optional game and map fields (may be None in emergency fallback)
  game_dict = (
      state_msg.game.model_dump(by_alias=False) if state_msg.game else {}
  )
  map_dict = state_msg.map.model_dump(by_alias=False) if state_msg.map else {}

  extracted = {
      "turn": state_msg.turn,
      "phase": state_msg.phase,
      "game": game_dict,
      "map": map_dict,
      "players": players_dicts,
      "units": units_dicts,
      "cities": cities_dicts,
  }

  # Include player_id if present (for agent perspective)
  if state_msg.player_id is not None:
      extracted["player_id"] = state_msg.player_id

  # Include legal_actions if present (for LLM agents)
  if hasattr(state_msg, 'legal_actions') and state_msg.legal_actions is not None:
      extracted["legal_actions"] = state_msg.legal_actions

  # Include LLM-optimized strategy fields if present
  # These are only sent when format="llm_optimized" from FreeCiv3D
  # TODO(AGE-196): Integrate strategic_summary into prompt builder for better LLM context
  # TODO(AGE-196): Use immediate_priorities to guide action selection in FreeCivLLMAgent
  # TODO(AGE-196): Incorporate threats into defensive action prioritization
  # TODO(AGE-196): Use opportunities to guide expansion and tech research decisions
  # See Linear issue: https://linear.app/agentclash/issue/AGE-196
  for field in ['strategic_summary', 'immediate_priorities', 'threats', 'opportunities']:
      if hasattr(state_msg, field) and getattr(state_msg, field) is not None:
          extracted[field] = getattr(state_msg, field)

  return extracted


def parse_message(message: Dict[str, Any]) -> FreeCiv3DMessage:
  """Parse any FreeCiv3D protocol message to appropriate model.

  Args:
    message: Raw message dict from WebSocket

  Returns:
    Appropriate FreeCiv3DMessage subclass instance

  Raises:
    ValidationError: If message doesn't match any known schema
    ValueError: If message type is unknown
  """
  msg_type = message.get("type")

  if msg_type == "state_update":
    return StateUpdateResponse.model_validate(message)
  elif msg_type == "auth_success":
    return AuthSuccessResponse.model_validate(message)
  elif msg_type == "error":
    return ErrorResponse.model_validate(message)
  elif msg_type == "action_result":
    return ActionResultResponse.model_validate(message)
  elif msg_type == "welcome":
    return WelcomeResponse.model_validate(message)
  elif msg_type == "game_ready":
    return GameReadyResponse.model_validate(message)
  elif msg_type == "llm_connect":
    return LLMConnectResponse.model_validate(message)
  else:
    # Unknown message type - return base model
    return FreeCiv3DMessage.model_validate(message)
