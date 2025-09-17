from __future__ import annotations

import json
import sys
from collections import OrderedDict
from dataclasses import replace
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Tuple)

from pydantic import BaseModel, Field, validator, model_validator

try:  # pragma: no cover - optional dependency
    import pyspiel  # type: ignore
except ImportError:  # pragma: no cover - fallback for tests without OpenSpiel
    pyspiel = None  # type: ignore


class _FallbackGameState:
    """Minimal fallback for pyspiel.State when OpenSpiel is unavailable.

    TODO: Explore disconnecting from pyspiel.State dependency.
    Consider creating a standalone GameState interface that doesn't require
    OpenSpiel compatibility. This would simplify the architecture and remove
    the need for this fallback mechanism.
    """

    def current_player(self) -> int:  # pragma: no cover - interface placeholder
        raise NotImplementedError

    def legal_actions(
        self, player: Optional[int] = None
    ) -> Sequence[int]:  # pragma: no cover
        raise NotImplementedError

    def is_terminal(self) -> bool:  # pragma: no cover
        raise NotImplementedError

    def returns(self) -> Sequence[float]:  # pragma: no cover
        raise NotImplementedError


_GameStateBase = pyspiel.State if pyspiel is not None else _FallbackGameState  # type: ignore

if pyspiel is not None:
    try:
        _PY_SPIEL_DUMMY_GAME = pyspiel.load_game('tic_tac_toe')
    except Exception:  # pragma: no cover - defensive fallback when OpenSpiel is misconfigured
        _PY_SPIEL_DUMMY_GAME = None
else:
    _PY_SPIEL_DUMMY_GAME = None

# Security and validation constants
MAX_STATE_SIZE_BYTES = 10_000_000  # 10MB limit for state data
MAX_PLAYER_ID = 1000
MAX_UNIT_ID = 100_000
MAX_CITY_ID = 100_000
MAX_TURN = 10_000
THREAT_DISTANCE_TILES = 3
LOW_HP_THRESHOLD = 50
MAX_JSON_DEPTH = 10
MAX_CACHE_SIZE = 100  # Maximum entries in LRU caches

# LLM observation constants
DEFAULT_MAX_TOKENS = 4000
MAX_PRIORITY_UNITS = 5
MAX_VISIBLE_ENEMY_UNITS = 10
MAX_ACTION_TYPES_DETAIL = 20
MAX_VISIBLE_RESOURCES = 5
MIN_TOKENS_FOR_ACTION_DETAILS = 1000

# Cache size constants for different cache types
ACTION_CACHE_SIZE = 50
OBSERVATION_CACHE_SIZE = 20
VISIBILITY_CACHE_SIZE = 30
THREAT_ANALYSIS_CACHE_SIZE = 25

# Detail level thresholds for observations
DETAIL_LEVEL_TOKENS = {
    'basic': 500,
    'medium': 1500,
    'full': 3000,
}


class LRUCache:
    """Simple LRU cache implementation with size limit."""

    def __init__(self, max_size: int = MAX_CACHE_SIZE):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()

    def get(self, key: Any, default: Any = None) -> Any:
        """Get value from cache, moving to end for LRU."""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return default

    def set(self, key: Any, value: Any) -> None:
        """Set value in cache, evicting oldest if necessary."""
        if key in self.cache:
            # Update existing key
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.max_size:
            # Remove oldest item
            self.cache.popitem(last=False)
        self.cache[key] = value

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()

    def __contains__(self, key: Any) -> bool:
        return key in self.cache

    def __len__(self) -> int:
        return len(self.cache)


def _calculate_deep_size(obj: Any, seen: Optional[set] = None) -> int:
    """Calculate the deep memory size of a nested object.

    Args:
        obj: Object to measure
        seen: Set of already seen objects to avoid infinite recursion

    Returns:
        Estimated deep size in bytes
    """
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0

    seen.add(obj_id)
    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        size += sum(_calculate_deep_size(k, seen) + _calculate_deep_size(v, seen)
                   for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set)):
        size += sum(_calculate_deep_size(item, seen) for item in obj)

    return size


def _validate_state_size(raw_state: Mapping[str, Any]) -> None:
    """Validate the size of raw state data to prevent DoS attacks.

    Args:
        raw_state: The raw state data to validate

    Raises:
        ValueError: If state data exceeds size limits
    """
    state_size = _calculate_deep_size(raw_state)
    if state_size > MAX_STATE_SIZE_BYTES:
        raise ValueError(
            f"State data exceeds maximum allowed size: {state_size} > {MAX_STATE_SIZE_BYTES}"
        )


def _safe_int_conversion(
    value: Any, max_value: int, field_name: str, allow_negative: bool = True
) -> int:
    """Safely convert value to int with bounds checking.

    Args:
        value: Value to convert to int
        max_value: Maximum allowed value
        field_name: Name of field for error messages
        allow_negative: Whether negative values are allowed

    Returns:
        The converted integer value

    Raises:
        ValueError: If value is invalid or out of bounds
    """
    # Try to convert to int if it's a string
    if isinstance(value, str):
        try:
            value = int(value)
        except ValueError:
            raise ValueError(f"Invalid {field_name}: cannot convert '{value}' to integer")
    elif not isinstance(value, int):
        raise ValueError(f"Invalid {field_name}: must be an integer, got {type(value).__name__}")

    if not allow_negative and value < 0:
        raise ValueError(f"Invalid {field_name}: negative values not allowed, got {value}")

    if value > max_value:
        raise ValueError(f"Invalid {field_name}: exceeds maximum value {max_value}, got {value}")

    return value


def _validate_state_structure(raw_state: Mapping[str, Any]) -> None:
    """Validate the basic structure of state data.

    Args:
        raw_state: The raw state data to validate

    Raises:
        TypeError: If required fields have wrong types
        ValueError: If required fields are missing
    """
    _validate_state_size(raw_state)

    required_fields = ['game', 'map', 'players', 'units', 'cities']
    for field in required_fields:
        if field not in raw_state:
            raise ValueError(f"Missing required field: {field}")

    # Validate field types
    if not isinstance(raw_state['game'], dict):
        raise TypeError("'game' field must be a dictionary")
    if not isinstance(raw_state['map'], dict):
        raise TypeError("'map' field must be a dictionary")
    if not isinstance(raw_state['players'], list):
        raise TypeError("'players' field must be a list")
    if not isinstance(raw_state['units'], list):
        raise TypeError("'units' field must be a list")
    if not isinstance(raw_state['cities'], list):
        raise TypeError("'cities' field must be a list")


def _safe_json_dumps(obj: Any, max_depth: int = MAX_JSON_DEPTH) -> str:
    """Safely serialize object to JSON with depth protection.

    Args:
        obj: Object to serialize
        max_depth: Maximum nesting depth allowed

    Returns:
        JSON string representation

    Raises:
        ValueError: If object is too deeply nested or contains circular references
    """

    def _check_depth(o: Any, current_depth: int = 0) -> None:
        if current_depth > max_depth:
            raise ValueError(f"Object nesting exceeds maximum depth of {max_depth}")

        if isinstance(o, dict):
            for value in o.values():
                _check_depth(value, current_depth + 1)
        elif isinstance(o, (list, tuple)):
            for item in o:
                _check_depth(item, current_depth + 1)

    # Check depth before serialization
    _check_depth(obj)

    try:
        return json.dumps(obj, sort_keys=True)
    except (TypeError, ValueError) as e:
        # Fallback to simple hash if JSON serialization fails
        return str(hash(str(obj)))


class FreeCivAction(BaseModel):
    """Represents a normalized FreeCiv action.

    Example:
        FreeCivAction(
            action_type="unit_move",
            actor_id=101,
            target={"x": 3, "y": 4},
            parameters={"direction": "NE"},
            source="unit",
        )
    """

    action_type: str = Field(..., min_length=1, max_length=50)
    actor_id: int = Field(..., ge=0, le=MAX_UNIT_ID)
    target: Optional[Dict[str, Any]] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    source: str = Field(..., pattern=r'^(unit|city|player)$')

    class Config:
        validate_assignment = True
        extra = "forbid"

    def to_packet(self) -> Dict[str, Any]:
        """Render the action into a protocol-friendly dictionary."""
        payload: Dict[str, Any] = {
            "action_type": self.action_type,
            "actor_id": self.actor_id,
            "parameters": self.parameters,
        }
        if self.target is not None:
            payload["target"] = self.target
        return payload


class FreeCivTile(BaseModel):
    """Represents a single tile on the FreeCiv map."""

    x: int = Field(..., ge=0, le=10000)  # Large maps can be up to 200x200
    y: int = Field(..., ge=0, le=10000)
    terrain: str = Field(..., min_length=1, max_length=20)
    resource: Optional[str] = Field(None, max_length=20)
    city_id: Optional[int] = Field(None, ge=0, le=MAX_CITY_ID)
    unit_ids: List[int] = Field(default_factory=list)
    improvements: List[str] = Field(default_factory=list)  # "road", "railroad", etc.
    pollution: bool = False
    fallout: bool = False
    owner: Optional[int] = Field(None, ge=0, le=MAX_PLAYER_ID)  # Territory ownership
    worked_by: Optional[int] = Field(None, ge=0, le=MAX_CITY_ID)  # City working this tile

    class Config:
        validate_assignment = True
        extra = "forbid"

    @validator('unit_ids', each_item=True)
    def validate_unit_ids(cls, v):
        if not (0 <= v <= MAX_UNIT_ID):
            raise ValueError(f'Unit ID {v} exceeds maximum {MAX_UNIT_ID}')
        return v

    @validator('improvements', each_item=True)
    def validate_improvements(cls, v):
        if len(v) > 30:  # Reasonable limit for improvement names
            raise ValueError(f'Improvement name too long: {v}')
        return v

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "terrain": self.terrain,
            "resource": self.resource,
            "city_id": self.city_id,
            "unit_ids": list(self.unit_ids),
            "improvements": list(self.improvements),
            "pollution": self.pollution,
            "fallout": self.fallout,
            "owner": self.owner,
            "worked_by": self.worked_by,
        }


class FreeCivPlayer(BaseModel):
    """Represents a FreeCiv player."""

    player_id: int = Field(..., ge=0, le=MAX_PLAYER_ID)
    name: str = Field(..., min_length=1, max_length=50)
    nation: str = Field(..., min_length=1, max_length=30)
    score: int = Field(..., ge=0)
    gold: int = Field(..., ge=0)
    techs: List[str] = Field(default_factory=list)
    government: Optional[str] = Field(None, max_length=20)
    science: int = Field(0, ge=0)
    research_target: Optional[str] = Field(None, max_length=30)
    research_progress: int = Field(0, ge=0, le=100)
    diplomatic_relations: Dict[int, str] = Field(default_factory=dict)
    trade_routes: List[Dict[str, Any]] = Field(default_factory=list)
    luxuries_rate: int = Field(0, ge=0, le=100)
    science_rate: int = Field(50, ge=0, le=100)
    tax_rate: int = Field(50, ge=0, le=100)

    class Config:
        validate_assignment = True
        extra = "forbid"

    @validator('techs', each_item=True)
    def validate_techs(cls, v):
        if len(v) > 50:  # Reasonable limit for tech names
            raise ValueError(f'Tech name too long: {v}')
        return v

    @validator('diplomatic_relations')
    def validate_diplomatic_relations(cls, v):
        valid_statuses = {"war", "peace", "ally", "ceasefire", "neutral"}
        for player_id, status in v.items():
            if not (0 <= player_id <= MAX_PLAYER_ID):
                raise ValueError(f'Invalid player ID in diplomatic relations: {player_id}')
            if status not in valid_statuses:
                raise ValueError(f'Invalid diplomatic status: {status}')
        return v

    @model_validator(mode='after')
    def validate_rates_sum(self):
        """Validate that tax, science, and luxury rates sum to 100."""
        tax = self.tax_rate
        science = self.science_rate
        luxury = self.luxuries_rate
        if tax + science + luxury != 100:
            raise ValueError(f'Tax, science, and luxury rates must sum to 100, got {tax + science + luxury}')
        return self


class FreeCivUnit(BaseModel):
    """Represents a FreeCiv unit."""

    unit_id: int = Field(..., ge=0, le=MAX_UNIT_ID)
    owner: int = Field(..., ge=0, le=MAX_PLAYER_ID)
    kind: str = Field(..., min_length=1, max_length=30)
    position: Tuple[int, int] = Field(...)
    hp: int = Field(..., ge=0, le=100)
    moves_left: int = Field(..., ge=0, le=10)
    veteran: bool = False
    orders: List[str] = Field(default_factory=list)
    available_actions: List[FreeCivAction] = Field(default_factory=list)
    fortified: bool = False
    activity: Optional[str] = None  # "exploring", "building_road", etc.
    fuel: int = Field(-1, ge=-1)  # -1 for unlimited, 0+ for air/naval units
    transport_id: Optional[int] = Field(None, ge=0, le=MAX_UNIT_ID)
    cargo_ids: List[int] = Field(default_factory=list)

    class Config:
        validate_assignment = True
        extra = "forbid"

    @validator('position')
    def validate_position(cls, v):
        x, y = v
        if not (0 <= x <= 10000 and 0 <= y <= 10000):
            raise ValueError(f'Position coordinates out of bounds: {v}')
        return v

    @validator('cargo_ids', each_item=True)
    def validate_cargo_ids(cls, v):
        if not (0 <= v <= MAX_UNIT_ID):
            raise ValueError(f'Cargo unit ID {v} exceeds maximum {MAX_UNIT_ID}')
        return v

    def move(self, x: int, y: int) -> None:
        self.position = (x, y)
        self.moves_left = max(self.moves_left - 1, 0)
        self.fortified = False  # Moving breaks fortification


class FreeCivCity(BaseModel):
    """Represents a FreeCiv city."""

    city_id: int = Field(..., ge=0, le=MAX_CITY_ID)
    owner: int = Field(..., ge=0, le=MAX_PLAYER_ID)
    name: str = Field(..., min_length=1, max_length=30)
    position: Tuple[int, int] = Field(...)
    population: int = Field(..., ge=1, le=50)  # Cities typically max at around 30-40
    production: Dict[str, Any] = Field(default_factory=dict)
    specialists: Dict[str, int] = Field(default_factory=dict)
    available_actions: List[FreeCivAction] = Field(default_factory=list)
    buildings: List[str] = Field(default_factory=list)
    food_storage: int = Field(0, ge=0)
    shield_storage: int = Field(0, ge=0)
    trade_routes: List[int] = Field(default_factory=list)  # IDs of connected cities
    under_siege: bool = False
    celebrating: bool = False
    disorder: bool = False
    worked_tiles: List[Tuple[int, int]] = Field(default_factory=list)

    class Config:
        validate_assignment = True
        extra = "forbid"

    @validator('position')
    def validate_position(cls, v):
        x, y = v
        if not (0 <= x <= 10000 and 0 <= y <= 10000):
            raise ValueError(f'Position coordinates out of bounds: {v}')
        return v

    @validator('buildings', each_item=True)
    def validate_buildings(cls, v):
        if len(v) > 50:  # Reasonable limit for building names
            raise ValueError(f'Building name too long: {v}')
        return v


class FreeCivMap(BaseModel):
    """Represents the FreeCiv game map."""

    width: int = Field(..., ge=1, le=10000)
    height: int = Field(..., ge=1, le=10000)
    tiles: Dict[Tuple[int, int], FreeCivTile] = Field(default_factory=dict)
    visibility: Dict[int, set[Tuple[int, int]]] = Field(default_factory=dict)

    class Config:
        validate_assignment = True
        extra = "forbid"
        arbitrary_types_allowed = True  # Allow set type in visibility

    @validator('visibility')
    def validate_visibility(cls, v):
        for player_id in v.keys():
            if not (0 <= player_id <= MAX_PLAYER_ID):
                raise ValueError(f'Invalid player ID in visibility: {player_id}')
        return v

    def visible_tiles(self, player_id: int) -> List[FreeCivTile]:
        coords = self.visibility.get(player_id, set())
        return [tile for coord, tile in self.tiles.items() if coord in coords]

    def move_unit(
        self, unit_id: int, from_pos: Tuple[int, int], to_pos: Tuple[int, int]
    ) -> None:
        if from_pos in self.tiles:
            self.tiles[from_pos].unit_ids = [
                uid for uid in self.tiles[from_pos].unit_ids if uid != unit_id
            ]
        if to_pos in self.tiles:
            tile = self.tiles[to_pos]
            if unit_id not in tile.unit_ids:
                tile.unit_ids.append(unit_id)
        else:
            # Create placeholder tile for unseen terrain to keep occupancy tracking coherent.
            self.tiles[to_pos] = FreeCivTile(
                x=to_pos[0],
                y=to_pos[1],
                terrain="unknown",
                resource=None,
                city_id=None,
                unit_ids=[unit_id],
            )


class FreeCivState(_GameStateBase):
    """Adapter translating FreeCiv JSON state into an OpenSpiel-like GameState."""

    def __init__(self, raw_state: Mapping[str, Any]):
        """Initialize FreeCiv state adapter with validation.

        Args:
            raw_state: Raw FreeCiv state data from JSON

        Raises:
            ValueError: If state data is invalid or exceeds size limits
            TypeError: If required fields have wrong types
        """
        if pyspiel is not None and _PY_SPIEL_DUMMY_GAME is not None:
            pyspiel.State.__init__(self, _PY_SPIEL_DUMMY_GAME)  # type: ignore[attr-defined]
        else:
            super().__init__()  # type: ignore[misc]

        # Validate input structure and size
        _validate_state_structure(raw_state)

        # Safely copy state data
        self._raw_state = dict(raw_state)
        self.game = dict(raw_state.get("game", {}))

        # Validate and parse game metadata with bounds checking
        self.turn: int = _safe_int_conversion(
            self.game.get("turn", 0), MAX_TURN, "turn"
        )
        self.phase: str = str(self.game.get("phase", "unknown"))
        self._is_terminal: bool = bool(self.game.get("is_over", False))

        # Parse scores with validation
        self._scores = self._parse_scores(self.game.get("scores", {}))

        # Validate current player
        current_player_raw = self.game.get("current_player", -1)
        if current_player_raw != -1:
            self._current_player_id = _safe_int_conversion(
                current_player_raw, MAX_PLAYER_ID, "current_player"
            )
        else:
            self._current_player_id = -1

        self.map = self._parse_map(raw_state.get("map", {}))
        self.players = self._parse_players(raw_state.get("players", []))
        self.units = self._parse_units(raw_state.get("units", []))
        self.cities = self._parse_cities(raw_state.get("cities", []))

        self._action_cache: LRUCache = LRUCache(max_size=ACTION_CACHE_SIZE)
        self._observation_cache: LRUCache = LRUCache(max_size=OBSERVATION_CACHE_SIZE)
        self._visibility_cache: LRUCache = LRUCache(max_size=VISIBILITY_CACHE_SIZE)
        self._threat_analysis_cache: LRUCache = LRUCache(max_size=THREAT_ANALYSIS_CACHE_SIZE)
        self._cache_version: int = 0  # Increment when state changes

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    def _parse_map(self, data: Mapping[str, Any]) -> FreeCivMap:
        width = int(data.get("width", 0) or 0)
        height = int(data.get("height", 0) or 0)
        tiles: Dict[Tuple[int, int], FreeCivTile] = {}
        for tile_data in data.get("tiles", []):
            x = int(tile_data.get("x", 0))
            y = int(tile_data.get("y", 0))
            tile = FreeCivTile(
                x=x,
                y=y,
                terrain=str(tile_data.get("terrain", "unknown")),
                resource=tile_data.get("resource"),
                city_id=tile_data.get("city_id"),
                unit_ids=list(tile_data.get("unit_ids", [])),
                improvements=list(tile_data.get("improvements", [])),
                pollution=bool(tile_data.get("pollution", False)),
                fallout=bool(tile_data.get("fallout", False)),
                owner=tile_data.get("owner"),
                worked_by=tile_data.get("worked_by"),
            )
            tiles[(x, y)] = tile
        visibility: Dict[int, set[Tuple[int, int]]] = {}
        for player_id, coords in data.get("visibility", {}).items():
            pid = int(player_id)
            visibility[pid] = {tuple(map(int, coord)) for coord in coords}
        return FreeCivMap(
            width=width, height=height, tiles=tiles, visibility=visibility
        )

    def _parse_scores(self, scores_data: Mapping[str, Any]) -> Dict[str, int]:
        """Parse and validate score data.

        Args:
            scores_data: Raw scores data from state

        Returns:
            Dictionary mapping player ID strings to scores

        Raises:
            TypeError: If scores_data is not a dictionary
            ValueError: If player IDs or scores are invalid
        """
        if not isinstance(scores_data, dict):
            raise TypeError("'scores' field must be a dictionary")

        scores: Dict[str, int] = {}
        for k, v in scores_data.items():
            player_id = _safe_int_conversion(k, MAX_PLAYER_ID, f"score player_id '{k}'")
            score_value = _safe_int_conversion(v, sys.maxsize, f"score for player {k}")
            scores[str(player_id)] = score_value
        return scores

    def _parse_players(
        self, players_data: Sequence[Mapping[str, Any]]
    ) -> Dict[int, FreeCivPlayer]:
        """Parse player data with validation.

        Args:
            players_data: Raw player data from state

        Returns:
            Dictionary mapping player ID to FreeCivPlayer objects

        Raises:
            ValueError: If player data is invalid
        """
        players: Dict[int, FreeCivPlayer] = {}
        for pdata in players_data:
            if not isinstance(pdata, dict):
                raise TypeError("Player data must be a dictionary")

            pid = _safe_int_conversion(pdata.get("id"), MAX_PLAYER_ID, "player_id")

            # Parse diplomatic relations with validation
            diplomatic_relations = {}
            for rel_data in pdata.get("diplomatic_relations", []):
                if not isinstance(rel_data, dict):
                    continue  # Skip invalid diplomatic data
                other_pid = _safe_int_conversion(
                    rel_data.get("player_id", 0), MAX_PLAYER_ID, "diplomatic_player_id"
                )
                relation = str(rel_data.get("status", "neutral"))
                diplomatic_relations[other_pid] = relation

            players[pid] = FreeCivPlayer(
                player_id=pid,
                name=str(pdata.get("name", f"Player {pid}")),
                nation=str(pdata.get("nation", "Unknown")),
                score=_safe_int_conversion(
                    pdata.get("score", 0), sys.maxsize, f"score for player {pid}"
                ),
                gold=_safe_int_conversion(
                    pdata.get("gold", 0), sys.maxsize, f"gold for player {pid}"
                ),
                techs=list(pdata.get("techs", [])),
                government=pdata.get("government"),
                science=_safe_int_conversion(
                    pdata.get("science", 0), sys.maxsize, f"science for player {pid}"
                ),
                research_target=pdata.get("research_target"),
                research_progress=_safe_int_conversion(
                    pdata.get("research_progress", 0),
                    100,
                    f"research_progress for player {pid}",
                ),
                diplomatic_relations=diplomatic_relations,
                trade_routes=list(pdata.get("trade_routes", [])),
                luxuries_rate=_safe_int_conversion(
                    pdata.get("luxuries_rate", 0),
                    100,
                    f"luxuries_rate for player {pid}",
                ),
                science_rate=_safe_int_conversion(
                    pdata.get("science_rate", 50), 100, f"science_rate for player {pid}"
                ),
                tax_rate=_safe_int_conversion(
                    pdata.get("tax_rate", 50), 100, f"tax_rate for player {pid}"
                ),
            )
        return players

    def _parse_units(
        self, units_data: Sequence[Mapping[str, Any]]
    ) -> Dict[int, FreeCivUnit]:
        """Parse unit data with validation.

        Args:
            units_data: Raw unit data from state

        Returns:
            Dictionary mapping unit ID to FreeCivUnit objects

        Raises:
            ValueError: If unit data is invalid
        """
        units: Dict[int, FreeCivUnit] = {}
        for udata in units_data:
            if not isinstance(udata, dict):
                raise TypeError("Unit data must be a dictionary")

            unit_id = _safe_int_conversion(udata.get("id"), MAX_UNIT_ID, "unit_id")
            position = (
                _safe_int_conversion(
                    udata.get("x", 0), 1000, f"unit {unit_id} x coordinate"
                ),
                _safe_int_conversion(
                    udata.get("y", 0), 1000, f"unit {unit_id} y coordinate"
                ),
            )

            actions = [
                self._convert_action(unit_id, action, "unit")
                for action in udata.get("available_actions", [])
                if isinstance(action, dict)
            ]

            # Validate transport_id if present
            transport_id = udata.get("transport_id")
            if transport_id is not None:
                transport_id = _safe_int_conversion(
                    transport_id, MAX_UNIT_ID, f"transport_id for unit {unit_id}"
                )

            units[unit_id] = FreeCivUnit(
                unit_id=unit_id,
                owner=_safe_int_conversion(
                    udata.get("owner"), MAX_PLAYER_ID, f"owner for unit {unit_id}"
                ),
                kind=str(udata.get("type", "unknown")),
                position=position,
                hp=_safe_int_conversion(
                    udata.get("hp", 0), 1000, f"hp for unit {unit_id}"
                ),
                moves_left=_safe_int_conversion(
                    udata.get("moves_left", 0), 100, f"moves_left for unit {unit_id}"
                ),
                veteran=bool(udata.get("veteran", False)),
                orders=list(udata.get("orders", [])),
                available_actions=actions,
                fortified=bool(udata.get("fortified", False)),
                activity=udata.get("activity"),
                fuel=_safe_int_conversion(
                    udata.get("fuel", -1),
                    1000,
                    f"fuel for unit {unit_id}",
                    allow_negative=True,
                ),
                transport_id=transport_id,
                cargo_ids=[
                    _safe_int_conversion(
                        cid, MAX_UNIT_ID, f"cargo_id for unit {unit_id}"
                    )
                    for cid in udata.get("cargo_ids", [])
                ],
            )
        return units

    def _parse_cities(
        self, cities_data: Sequence[Mapping[str, Any]]
    ) -> Dict[int, FreeCivCity]:
        """Parse city data with validation.

        Args:
            cities_data: Raw city data from state

        Returns:
            Dictionary mapping city ID to FreeCivCity objects

        Raises:
            ValueError: If city data is invalid
        """
        cities: Dict[int, FreeCivCity] = {}
        for cdata in cities_data:
            if not isinstance(cdata, dict):
                raise TypeError("City data must be a dictionary")

            city_id = _safe_int_conversion(cdata.get("id"), MAX_CITY_ID, "city_id")
            position = (
                _safe_int_conversion(
                    cdata.get("x", 0), 1000, f"city {city_id} x coordinate"
                ),
                _safe_int_conversion(
                    cdata.get("y", 0), 1000, f"city {city_id} y coordinate"
                ),
            )

            actions = [
                self._convert_action(city_id, action, "city")
                for action in cdata.get("available_actions", [])
                if isinstance(action, dict)
            ]

            # Parse worked tiles with validation
            worked_tiles = []
            for tile_coords in cdata.get("worked_tiles", []):
                if isinstance(tile_coords, (list, tuple)) and len(tile_coords) >= 2:
                    try:
                        x = _safe_int_conversion(
                            tile_coords[0], 1000, f"worked_tile x for city {city_id}"
                        )
                        y = _safe_int_conversion(
                            tile_coords[1], 1000, f"worked_tile y for city {city_id}"
                        )
                        worked_tiles.append((x, y))
                    except (ValueError, TypeError):
                        continue  # Skip invalid tile coordinates

            # Parse trade routes with validation
            trade_routes = []
            for trade_route in cdata.get("trade_routes", []):
                if isinstance(trade_route, (int, str)):
                    try:
                        route_id = _safe_int_conversion(
                            trade_route, MAX_CITY_ID, f"trade_route for city {city_id}"
                        )
                        trade_routes.append(route_id)
                    except (ValueError, TypeError):
                        continue  # Skip invalid trade route IDs

            cities[city_id] = FreeCivCity(
                city_id=city_id,
                owner=_safe_int_conversion(
                    cdata.get("owner"), MAX_PLAYER_ID, f"owner for city {city_id}"
                ),
                name=str(cdata.get("name", f"City {city_id}")),
                position=position,
                population=_safe_int_conversion(
                    cdata.get("population", 0), 1000, f"population for city {city_id}"
                ),
                production=dict(cdata.get("production", {})),
                specialists=dict(cdata.get("specialists", {})),
                available_actions=actions,
                buildings=list(cdata.get("buildings", [])),
                food_storage=_safe_int_conversion(
                    cdata.get("food_storage", 0),
                    10000,
                    f"food_storage for city {city_id}",
                ),
                shield_storage=_safe_int_conversion(
                    cdata.get("shield_storage", 0),
                    10000,
                    f"shield_storage for city {city_id}",
                ),
                trade_routes=trade_routes,
                under_siege=bool(cdata.get("under_siege", False)),
                celebrating=bool(cdata.get("celebrating", False)),
                disorder=bool(cdata.get("disorder", False)),
                worked_tiles=worked_tiles,
            )
        return cities

    def _convert_action(
        self, actor_id: int, action_data: Mapping[str, Any], source: str
    ) -> FreeCivAction:
        action_type = str(action_data.get("type", "unknown"))
        target_raw = action_data.get("target") or action_data.get("target_unit")
        if isinstance(target_raw, Mapping):
            target = {key: value for key, value in target_raw.items()}
        elif target_raw is None:
            target = None
        else:
            target = {"id": target_raw}
        parameters = dict(action_data.get("parameters", {}))
        return FreeCivAction(
            action_type=action_type,
            actor_id=actor_id,
            target=target,
            parameters=parameters,
            source=source,
        )

    # ------------------------------------------------------------------
    # OpenSpiel-like interface
    # ------------------------------------------------------------------
    def current_player(self) -> int:
        if self._current_player_id < 0:
            return -1
        return self._current_player_id - 1

    def legal_actions(
        self, player: Optional[int] = None
    ) -> Sequence[int]:  # pragma: no cover - compatibility hook
        player_id = self._current_player_id if player is None else player + 1
        legal = self.get_legal_actions(player_id)
        cache_key = (player_id, self._cache_version)
        self._action_cache.set(cache_key, list(legal))
        return list(range(len(legal)))

    def action_to_string(self, player: int, action_id: int) -> str:
        """Convert action index to human-readable string."""
        player_id = player + 1 if player >= 0 else self._current_player_id
        cache_key = (player_id, self._cache_version)
        cached_actions = self._action_cache.get(cache_key)
        if cached_actions is None:
            self.get_legal_actions(player_id)
            cached_actions = self._action_cache.get(cache_key)

        if action_id >= len(cached_actions):
            raise ValueError(
                f"Action index {action_id} out of bounds for player {player_id}"
            )

        action = cached_actions[action_id]
        return self._action_to_string(action)

    def string_to_action(self, player: int, action_str: str) -> int:
        """Convert human-readable string to action index."""
        player_id = player + 1 if player >= 0 else self._current_player_id
        cache_key = (player_id, self._cache_version)
        cached_actions = self._action_cache.get(cache_key)
        if cached_actions is None:
            self.get_legal_actions(player_id)
            cached_actions = self._action_cache.get(cache_key)

        for i, action in enumerate(cached_actions):
            if self._action_to_string(action) == action_str:
                return i

        raise ValueError(
            f"Action string '{action_str}' not found for player {player_id}"
        )

    def apply_action_by_index(self, action_id: int) -> None:
        """Apply action by OpenSpiel action index."""
        player_id = self._current_player_id
        cache_key = (player_id, self._cache_version)
        cached_actions = self._action_cache.get(cache_key)
        if cached_actions is None:
            self.get_legal_actions(player_id)
            cached_actions = self._action_cache.get(cache_key)

        if action_id >= len(cached_actions):
            raise ValueError(f"Action index {action_id} out of bounds")

        action = cached_actions[action_id]
        self.apply_action(action)

    def is_terminal(self) -> bool:
        return self._is_terminal

    def returns(self) -> List[float]:
        if not self._scores:
            return []
        ordered_scores = []
        for player_id in sorted(self.players):
            ordered_scores.append(float(self._scores.get(str(player_id), 0)))
        return ordered_scores

    def __str__(self) -> str:
        return f"FreeCivState(turn={self.turn}, phase={self.phase}, current_player={self._current_player_id})"

    # ------------------------------------------------------------------
    # Adapter functionality
    # ------------------------------------------------------------------
    def get_legal_actions(self, player_id: int) -> List[FreeCivAction]:
        if player_id not in self.players:
            return []
        actions: List[FreeCivAction] = []
        for unit in self.units.values():
            if unit.owner == player_id:
                actions.extend(unit.available_actions)
        for city in self.cities.values():
            if city.owner == player_id:
                actions.extend(city.available_actions)
        # Deduplicate actions by type/target to avoid duplicates in fixtures.
        deduped: Dict[str, FreeCivAction] = {}
        for action in actions:
            key = _safe_json_dumps(
                {
                    "type": action.action_type,
                    "actor": action.actor_id,
                    "target": action.target,
                }
            )
            deduped[key] = action
        result = list(deduped.values())
        cache_key = (player_id, self._cache_version)
        self._action_cache.set(cache_key, result)
        return [action.model_copy() for action in result]

    def to_observation(self, player_id: int, format: str = "enhanced") -> Any:
        cache_key = (player_id, format, self._cache_version)
        cached_result = self._observation_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        if format == "enhanced":
            observation = self._build_llm_observation(player_id)
        elif format == "json":
            observation = self._build_json_observation(player_id)
        elif format == "ascii":
            observation = self._build_ascii_observation(player_id)
        else:
            raise ValueError(f"Unsupported observation format: {format}")

        self._observation_cache.set(cache_key, observation)
        return observation

    def apply_action(self, action: FreeCivAction) -> None:
        if action.source == "unit":
            self._apply_unit_action(action)
            self._invalidate_caches(
                "units"
            )  # Unit actions affect units, actions, and threats
        elif action.source == "city":
            self._apply_city_action(action)
            self._invalidate_caches("cities")  # City actions affect cities and actions
        else:  # pragma: no cover - defensive guard for unexpected data
            raise ValueError(f"Unknown action source: {action.source}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _invalidate_caches(self, scope: str = "all") -> None:
        """Selectively invalidate caches based on change scope.

        Args:
            scope: The scope of invalidation ("all", "visibility", "actions", "threats")
        """
        if scope == "all":
            self._action_cache.clear()
            self._observation_cache.clear()
            self._visibility_cache.clear()
            self._threat_analysis_cache.clear()
        elif scope == "visibility":
            self._visibility_cache.clear()
            self._observation_cache.clear()  # Observations depend on visibility
        elif scope == "actions":
            self._action_cache.clear()
            self._observation_cache.clear()  # Observations include action counts
        elif scope == "threats":
            self._threat_analysis_cache.clear()
            self._observation_cache.clear()  # Observations include threat data
        elif scope == "units":
            # Unit changes affect actions, threats, and observations
            self._action_cache.clear()
            self._threat_analysis_cache.clear()
            self._observation_cache.clear()
        elif scope == "cities":
            # City changes affect actions and observations
            self._action_cache.clear()
            self._observation_cache.clear()

        self._cache_version += 1

    def _get_cached_visibility(self, player_id: int) -> set[Tuple[int, int]]:
        """Get cached visibility for player to avoid recomputing."""
        cache_key = (player_id, self._cache_version)
        cached_result = self._visibility_cache.get(cache_key)
        if cached_result is None:
            visibility = self.map.visibility.get(player_id, set())
            self._visibility_cache.set(cache_key, visibility)
            return visibility
        return cached_result

    def _get_threat_analysis(self, player_id: int) -> Dict[str, Any]:
        """Get cached threat analysis for performance."""
        cache_key = (player_id, self._cache_version)
        cached_result = self._threat_analysis_cache.get(cache_key)
        if cached_result is None:
            friendly_units = [u for u in self.units.values() if u.owner == player_id]
            enemy_units = [u for u in self.units.values() if u.owner != player_id]

            threat_count = 0
            units_at_risk = 0
            for friendly in friendly_units:
                if friendly.hp < LOW_HP_THRESHOLD:
                    units_at_risk += 1
                for enemy in enemy_units:
                    distance = abs(friendly.position[0] - enemy.position[0]) + abs(
                        friendly.position[1] - enemy.position[1]
                    )
                    if distance <= THREAT_DISTANCE_TILES:
                        threat_count += 1
                        break

            threat_analysis = {
                "threat_count": threat_count,
                "units_at_risk": units_at_risk,
                "friendly_count": len(friendly_units),
                "enemy_count": len(enemy_units),
            }
            self._threat_analysis_cache.set(cache_key, threat_analysis)
            return threat_analysis

        return cached_result

    def _action_to_string(self, action: FreeCivAction) -> str:
        """Convert FreeCivAction to human-readable string."""
        parts = [action.action_type]

        if action.source == "unit" and action.actor_id in self.units:
            unit = self.units[action.actor_id]
            parts.append(f"{unit.kind}({action.actor_id})")
        elif action.source == "city" and action.actor_id in self.cities:
            city = self.cities[action.actor_id]
            parts.append(f"{city.name}({action.actor_id})")
        else:
            parts.append(f"actor({action.actor_id})")

        if action.target:
            if "x" in action.target and "y" in action.target:
                parts.append(f"to({action.target['x']},{action.target['y']})")
            elif "id" in action.target:
                parts.append(f"target({action.target['id']})")
            elif isinstance(action.target, str):
                parts.append(f"target({action.target})")
            elif isinstance(action.target, dict) and "value" in action.target:
                parts.append(f"target({action.target['value']})")

        if action.parameters:
            key_params = ["direction", "damage", "focus", "priority"]
            for key in key_params:
                if key in action.parameters:
                    parts.append(f"{key}({action.parameters[key]})")

        return "_".join(parts)

    def _apply_unit_action(self, action: FreeCivAction) -> None:
        """Apply unit action with detailed error context."""
        unit = self.units.get(action.actor_id)
        if unit is None:
            available_units = list(self.units.keys())[:10]  # Show up to 10 for context
            raise ValueError(
                f"Unit {action.actor_id} not found for action '{action.action_type}'. "
                f"Available units: {available_units}"
            )

        if action.action_type == "unit_move" and action.target:
            destination = (
                int(action.target.get("x", unit.position[0])),
                int(action.target.get("y", unit.position[1])),
            )
            self.map.move_unit(unit.unit_id, unit.position, destination)
            unit.move(*destination)
        elif action.action_type == "unit_attack" and action.target:
            target_id = action.target.get("id")
            if target_id is not None and target_id in self.units:
                defender = self.units[target_id]
                defender.hp = max(
                    defender.hp - int(action.parameters.get("damage", 10)), 0
                )
                if defender.hp == 0:
                    self._remove_unit(defender.unit_id)
        elif action.action_type == "unit_fortify":
            unit.fortified = True
            unit.moves_left = 0
            unit.activity = None
        elif action.action_type == "unit_explore":
            unit.activity = "exploring"
            unit.orders = ["exploring"]
        elif action.action_type == "unit_build_improvement" and action.target:
            improvement = action.parameters.get("improvement", "road")
            target_pos = (
                int(action.target.get("x", unit.position[0])),
                int(action.target.get("y", unit.position[1])),
            )
            if target_pos in self.map.tiles:
                tile = self.map.tiles[target_pos]
                if improvement not in tile.improvements:
                    tile.improvements.append(improvement)
            unit.activity = f"building_{improvement}"
            unit.moves_left = 0
        elif action.action_type in {"unit_road", "unit_irrigation", "unit_mine"}:
            improvement_map = {
                "unit_road": "road",
                "unit_irrigation": "irrigation",
                "unit_mine": "mine",
            }
            improvement = improvement_map[action.action_type]
            target_pos = action.target if action.target else unit.position
            if isinstance(target_pos, dict):
                target_pos = (
                    int(target_pos.get("x", unit.position[0])),
                    int(target_pos.get("y", unit.position[1])),
                )
            if target_pos in self.map.tiles:
                tile = self.map.tiles[target_pos]
                if improvement not in tile.improvements:
                    tile.improvements.append(improvement)
            unit.activity = f"building_{improvement}"
            unit.moves_left = 0
        elif action.action_type == "unit_transport" and action.target:
            cargo_id = action.target.get("id")
            if cargo_id and cargo_id in self.units and cargo_id not in unit.cargo_ids:
                unit.cargo_ids.append(cargo_id)
                self.units[cargo_id].transport_id = unit.unit_id
        elif action.action_type == "unit_unload" and action.target:
            cargo_id = action.target.get("id")
            if cargo_id and cargo_id in unit.cargo_ids:
                unit.cargo_ids.remove(cargo_id)
                if cargo_id in self.units:
                    self.units[cargo_id].transport_id = None
        elif action.action_type.startswith("unit_"):
            # Generic state mutation placeholder: track orders.
            if action.action_type not in unit.orders:
                unit.orders.append(action.action_type)
        else:  # pragma: no cover - unexpected action type
            raise ValueError(f"Unsupported unit action: {action.action_type}")

    def _remove_unit(self, unit_id: int) -> None:
        unit = self.units.pop(unit_id, None)
        if not unit:
            return
        tile = self.map.tiles.get(unit.position)
        if tile:
            tile.unit_ids = [uid for uid in tile.unit_ids if uid != unit_id]

    def _apply_city_action(self, action: FreeCivAction) -> None:
        """Apply city action with detailed error context."""
        city = self.cities.get(action.actor_id)
        if city is None:
            available_cities = list(self.cities.keys())[
                :10
            ]  # Show up to 10 for context
            raise ValueError(
                f"City {action.actor_id} not found for action '{action.action_type}'. "
                f"Available cities: {available_cities}"
            )

        if action.action_type in {"city_production", "city_switch_production"}:
            target = (
                action.target
                if isinstance(action.target, dict)
                else {"value": action.target}
            )
            if target:
                city.production["current"] = (
                    target.get("value")
                    or target.get("name")
                    or target.get("focus")
                    or target
                )
        elif action.action_type == "city_adjust_specialist" and action.target:
            for specialist, value in action.target.items():
                city.specialists[specialist] = int(value)
        elif action.action_type == "city_build_improvement" and action.target:
            improvement = (
                action.target.get("value") or action.target.get("name") or action.target
            )
            if isinstance(improvement, str) and improvement not in city.buildings:
                city.buildings.append(improvement)
        elif action.action_type == "city_set_citizens" and action.target:
            # Handle citizen assignment to worked tiles
            for tile_coord, worked in action.target.items():
                coords = tuple(map(int, tile_coord.split(",")))
                if worked and coords not in city.worked_tiles:
                    city.worked_tiles.append(coords)
                elif not worked and coords in city.worked_tiles:
                    city.worked_tiles.remove(coords)
        elif action.action_type == "city_set_tax_rates" and action.target:
            # This would typically affect the player, not the city
            player = self.players.get(city.owner)
            if player and action.target:
                player.tax_rate = int(action.target.get("tax", player.tax_rate))
                player.science_rate = int(
                    action.target.get("science", player.science_rate)
                )
                player.luxuries_rate = int(
                    action.target.get("luxuries", player.luxuries_rate)
                )
        elif action.action_type == "city_establish_trade_route" and action.target:
            target_city_id = action.target.get("id")
            if target_city_id and target_city_id not in city.trade_routes:
                city.trade_routes.append(target_city_id)
        elif action.action_type == "city_celebrate":
            city.celebrating = True
            city.disorder = False
        elif action.action_type == "city_quell_disorder":
            city.disorder = False
        elif action.action_type.startswith("city_"):
            # Record the most recent city directive for prompt building context.
            city.production.setdefault("directives", []).append(action.action_type)
        else:  # pragma: no cover - unexpected type
            raise ValueError(f"Unsupported city action: {action.action_type}")

    def _build_json_observation(self, player_id: int) -> Dict[str, Any]:
        visible_coords = self._get_cached_visibility(player_id)
        visible_tiles = [
            tile.to_dict()
            for coord, tile in self.map.tiles.items()
            if coord in visible_coords
        ]
        units = self._get_visible_units(player_id)
        cities = self._get_visible_cities(player_id)
        observation = {
            "game": {"turn": self.turn, "phase": self.phase, "player_id": player_id},
            "map": {
                "visible_tiles": visible_tiles,
                "width": self.map.width,
                "height": self.map.height,
            },
            "units": units,
            "cities": cities,
            "legal_actions": [
                action.to_packet() for action in self.get_legal_actions(player_id)
            ],
        }
        return observation

    def _build_ascii_observation(self, player_id: int) -> str:
        visible = self._get_cached_visibility(player_id)
        rows: List[str] = []
        for y in range(self.map.height):
            chars: List[str] = []
            for x in range(self.map.width):
                coord = (x, y)
                tile = self.map.tiles.get(coord)
                if coord not in visible or tile is None:
                    chars.append("?")
                    continue
                char = self._terrain_symbol(tile.terrain)
                if tile.city_id is not None and tile.city_id in self.cities:
                    owner = self.cities[tile.city_id].owner
                    char = "C" if owner == player_id else "c"
                elif tile.unit_ids:
                    # Preference for friendly units when both sides occupy a tile.
                    owner = None
                    for unit_id in tile.unit_ids:
                        unit = self.units.get(unit_id)
                        unit_owner = unit.owner if unit else None
                        if unit_owner == player_id:
                            owner = unit_owner
                            break
                        owner = unit_owner
                    char = "U" if owner == player_id else "u"
                chars.append(char)
            rows.append("".join(chars))
        return "\n".join(rows)

    def _terrain_symbol(self, terrain: str) -> str:
        terrain = terrain.lower()
        if terrain in {"ocean", "coast"}:
            return "~"
        if terrain in {"river"}:
            return "="
        if terrain in {"hill", "mountain"}:
            return "^"
        if terrain in {"forest"}:
            return "F"
        if terrain in {"city"}:
            return "#"
        return "."

    def _build_llm_observation(
        self, player_id: int, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> Dict[str, Any]:
        """Build LLM-optimized observation with adaptive detail levels.

        Creates a hierarchical observation structure optimized for LLM processing
        within token constraints. Uses adaptive detail levels based on available
        token budget to ensure the most important information is always included.

        Args:
            player_id: The player to build observation for
            max_tokens: Maximum token budget for the observation (default: 4000)

        Returns:
            Dictionary containing:
            - metadata: Basic game state info (turn, phase, player_id)
            - strategic: High-level strategic information (research, economy, diplomacy)
            - tactical: Unit and threat information (adaptive based on token budget)
            - economic: City and resource information (if token budget allows)
            - actions: Legal action summary and types
            - metrics: Token usage and detail level metrics

        Example output structure::

            {
                "metadata": {"turn": 5, "phase": "movement"},
                "strategic": {"scoreboard": {"player": 42}},
                "tactical": {"unit_counts": {"friendly": 3, "enemy": 2}},
                "economic": {"cities": [...]},
                "actions": {"legal": ["unit_move", "city_production"]},
                "metrics": {"estimated_tokens": 850, "detail_level": "medium"}
            }

        Note:
            Uses cached threat analysis and visibility data for performance.
            Token estimation is approximate (4 chars ≈ 1 token for JSON).
        """
        player = self.players.get(player_id)
        units = self._get_visible_units(player_id)
        enemy_units = [unit for unit in units if unit["owner"] != player_id]
        friendly_units = [unit for unit in units if unit["owner"] == player_id]
        cities = self._get_visible_cities(player_id)

        # Build core strategic information first (always included)
        core_observation = self._build_core_strategic_observation(player_id, player)

        # Add tactical summary (condensed) using cached threat analysis
        tactical_summary = self._build_tactical_summary(player_id, cities)

        # Estimate token usage so far
        base_size = self._estimate_token_count(
            core_observation
        ) + self._estimate_token_count(tactical_summary)

        # Adaptive detail level based on remaining token budget
        remaining_tokens = max_tokens - base_size
        detail_level = self._determine_detail_level(remaining_tokens)

        # Add detailed sections based on available space
        if detail_level >= 1:  # Basic details
            core_observation["tactical"] = tactical_summary
            if detail_level >= 2:  # More details
                core_observation["tactical"]["priority_units"] = (
                    self._get_priority_units(friendly_units, enemy_units)[:MAX_PRIORITY_UNITS]
                )
                if detail_level >= 3:  # Full details
                    core_observation["tactical"]["visible_enemy_units"] = enemy_units[
                        :MAX_VISIBLE_ENEMY_UNITS
                    ]
                    core_observation["economic"] = {
                        "cities": self._compress_city_data(cities, player_id),
                        "resources": self._extract_visible_resources(player_id)[
                            :MAX_VISIBLE_RESOURCES
                        ],
                    }

        # Add legal actions summary (always include count, details if space allows)
        legal_actions = self.get_legal_actions(player_id)
        core_observation["actions"] = {"count": len(legal_actions)}

        if remaining_tokens > MIN_TOKENS_FOR_ACTION_DETAILS:  # Include action details if space permits
            action_types = {}
            for action in legal_actions[:MAX_ACTION_TYPES_DETAIL]:
                action_types[action.action_type] = (
                    action_types.get(action.action_type, 0) + 1
                )
            core_observation["actions"]["types"] = action_types

        # Final metrics
        observation_str = json.dumps(core_observation)
        token_count = self._estimate_token_count(core_observation)

        core_observation["metrics"] = {
            "char_count": len(observation_str),
            "estimated_tokens": token_count,
            "detail_level": detail_level,
            "truncated": token_count > max_tokens,
        }

        return core_observation

    def _build_core_strategic_observation(
        self, player_id: int, player: Optional[FreeCivPlayer]
    ) -> Dict[str, Any]:
        """Build core strategic observation data that's always included.

        Args:
            player_id: The player to build observation for
            player: The player object (may be None)

        Returns:
            Dictionary containing metadata and strategic information
        """
        return {
            "metadata": {
                "turn": self.turn,
                "phase": self.phase,
                "player_id": player_id,
            },
            "strategic": {
                "turn": self.turn,
                "phase": self.phase,
                "government": player.government if player else None,
                "scoreboard": {
                    "player": self._scores.get(str(player_id), 0),
                    "opponents": [
                        score
                        for pid, score in self._scores.items()
                        if int(pid) != player_id
                    ],
                },
                "research": (
                    {
                        "target": player.research_target if player else None,
                        "progress": player.research_progress if player else 0,
                        "completed_techs": len(player.techs) if player else 0,
                    }
                    if player
                    else None
                ),
                "economy": (
                    {
                        "gold": player.gold if player else 0,
                        "science_rate": player.science_rate if player else 50,
                        "tax_rate": player.tax_rate if player else 50,
                    }
                    if player
                    else None
                ),
            },
        }

    def _build_tactical_summary(
        self, player_id: int, cities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build tactical summary using cached threat analysis.

        Args:
            player_id: The player to build tactical summary for
            cities: List of visible cities

        Returns:
            Dictionary containing tactical summary information
        """
        threat_data = self._get_threat_analysis(player_id)
        return {
            "unit_counts": {
                "friendly": threat_data["friendly_count"],
                "enemy": threat_data["enemy_count"],
                "at_risk": threat_data["units_at_risk"],
            },
            "city_count": len([c for c in cities if c["owner"] == player_id]),
            "threats": threat_data["threat_count"],
        }

    def _get_visible_units(self, player_id: int) -> List[Dict[str, Any]]:
        visible_coords = self._get_cached_visibility(player_id)
        payload: List[Dict[str, Any]] = []
        for unit in self.units.values():
            if unit.owner == player_id or unit.position in visible_coords:
                payload.append(
                    {
                        "id": unit.unit_id,
                        "owner": unit.owner,
                        "type": unit.kind,
                        "x": unit.position[0],
                        "y": unit.position[1],
                        "hp": unit.hp,
                        "moves_left": unit.moves_left,
                    }
                )
        return payload

    def _get_visible_cities(self, player_id: int) -> List[Dict[str, Any]]:
        visible_coords = self._get_cached_visibility(player_id)
        payload: List[Dict[str, Any]] = []
        for city in self.cities.values():
            if city.owner == player_id or city.position in visible_coords:
                payload.append(
                    {
                        "id": city.city_id,
                        "owner": city.owner,
                        "name": city.name,
                        "x": city.position[0],
                        "y": city.position[1],
                        "population": city.population,
                        "production": city.production,
                        "specialists": city.specialists,
                    }
                )
        return payload

    def _extract_visible_resources(self, player_id: int) -> List[Dict[str, Any]]:
        visible_tiles = self.map.visible_tiles(player_id)
        return [
            {"x": tile.x, "y": tile.y, "resource": tile.resource}
            for tile in visible_tiles
            if tile.resource
        ]

    def _estimate_token_count(self, data: Any) -> int:
        """Rough estimate of token count for given data structure."""
        json_str = json.dumps(data)
        # Rough approximation: 1 token ≈ 4 characters for JSON
        return len(json_str) // 4

    def _determine_detail_level(self, remaining_tokens: int) -> int:
        """Determine level of detail based on remaining token budget."""
        if remaining_tokens > DETAIL_LEVEL_TOKENS['full']:
            return 3  # Full details
        elif remaining_tokens > DETAIL_LEVEL_TOKENS['medium']:
            return 2  # Medium details
        elif remaining_tokens > DETAIL_LEVEL_TOKENS['basic']:
            return 1  # Basic details
        else:
            return 0  # Minimal details

    def _is_unit_threatening(
        self, enemy_unit: Dict[str, Any], friendly_units: List[Dict[str, Any]]
    ) -> bool:
        """Check if an enemy unit poses a threat to friendly units."""
        enemy_pos = (enemy_unit["x"], enemy_unit["y"])
        for friendly in friendly_units:
            friendly_pos = (friendly["x"], friendly["y"])
            # Simple distance check (Manhattan distance)
            distance = abs(enemy_pos[0] - friendly_pos[0]) + abs(
                enemy_pos[1] - friendly_pos[1]
            )
            if distance <= THREAT_DISTANCE_TILES:
                return True
        return False

    def _get_priority_units(
        self, friendly_units: List[Dict[str, Any]], enemy_units: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get priority units based on strategic importance."""
        priority_units = []

        # Add units that are at risk
        at_risk = [
            unit for unit in friendly_units if unit.get("hp", 0) < LOW_HP_THRESHOLD
        ]
        priority_units.extend(at_risk)

        # Add units near enemies
        for unit in friendly_units:
            if self._is_unit_threatening(unit, enemy_units):
                priority_units.append(unit)

        # Add units with moves remaining
        active_units = [
            unit for unit in friendly_units if unit.get("moves_left", 0) > 0
        ]
        priority_units.extend(active_units[:3])  # Top 3 active units

        # Remove duplicates while preserving order
        seen = set()
        result = []
        for unit in priority_units:
            unit_id = unit["id"]
            if unit_id not in seen:
                seen.add(unit_id)
                result.append(unit)

        return result

    def _compress_city_data(
        self, cities: List[Dict[str, Any]], player_id: int
    ) -> List[Dict[str, Any]]:
        """Compress city data for token efficiency."""
        compressed = []
        for city in cities:
            if city["owner"] == player_id:
                # Full details for own cities
                compressed.append(
                    {
                        "id": city["id"],
                        "name": city["name"],
                        "pop": city["population"],
                        "production": city.get("production", {}).get(
                            "current", "unknown"
                        ),
                        "x": city["x"],
                        "y": city["y"],
                    }
                )
            else:
                # Minimal details for enemy cities
                compressed.append(
                    {
                        "id": city["id"],
                        "name": city["name"],
                        "pop": city["population"],
                        "x": city["x"],
                        "y": city["y"],
                        "enemy": True,
                    }
                )
        return compressed


__all__ = [
    "FreeCivAction",
    "FreeCivCity",
    "FreeCivMap",
    "FreeCivState",
    "FreeCivTile",
    "FreeCivUnit",
    "FreeCivPlayer",
]
