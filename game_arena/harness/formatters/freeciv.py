# Copyright 2025 The game_arena Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom formatter for FreeCiv game states."""

import json
from typing import Any, Dict, List

from game_arena.harness.freeciv_state import FreeCivState


def _player_string(player_id: int) -> str:
    """Convert player ID to readable string."""
    if player_id < 0:
        return "observer"
    elif player_id == 0:
        return "neutral"
    else:
        return f"Player {player_id}"


def _format_map_grid(
    freeciv_state: FreeCivState, player_id: int
) -> List[List[Dict[str, Any]]]:
    """Convert FreeCiv map to structured grid format.

    Args:
        freeciv_state: The FreeCiv game state
        player_id: Player ID for visibility filtering

    Returns:
        2D grid of map tiles with visibility considered
    """
    visible_coords = freeciv_state._get_cached_visibility(player_id)
    grid = []

    for y in range(freeciv_state.map.height):
        row = []
        for x in range(freeciv_state.map.width):
            coord = (x, y)
            if coord in visible_coords and coord in freeciv_state.map.tiles:
                tile = freeciv_state.map.tiles[coord]
                tile_info = {
                    "coordinate": f"{x},{y}",
                    "terrain": tile.terrain,
                    "resource": tile.resource,
                    "improvements": tile.improvements,
                    "pollution": tile.pollution,
                    "fallout": tile.fallout,
                    "owner": tile.owner,
                    "worked_by": tile.worked_by,
                    "units": tile.unit_ids,
                    "city": tile.city_id,
                }
            else:
                # Fog of war
                tile_info = {
                    "coordinate": f"{x},{y}",
                    "terrain": "unknown",
                    "visible": False,
                }
            row.append(tile_info)
        grid.append(row)

    return grid


def _format_units_summary(
    freeciv_state: FreeCivState, player_id: int
) -> List[Dict[str, Any]]:
    """Format units visible to player.

    Args:
        freeciv_state: The FreeCiv game state
        player_id: Player ID for visibility filtering

    Returns:
        List of formatted unit information
    """
    visible_coords = freeciv_state._get_cached_visibility(player_id)
    formatted_units = []

    for unit in freeciv_state.units.values():
        # Include own units and visible enemy units
        if unit.owner == player_id or unit.position in visible_coords:
            unit_info = {
                "id": unit.unit_id,
                "type": unit.kind,
                "position": {"x": unit.position[0], "y": unit.position[1]},
                "owner": unit.owner,
                "hp": unit.hp,
                "moves_left": unit.moves_left,
                "veteran": unit.veteran,
                "fortified": unit.fortified,
                "activity": unit.activity,
                "available_actions": (
                    len(unit.available_actions) if unit.owner == player_id else 0
                ),
            }
            formatted_units.append(unit_info)

    return formatted_units


def _format_cities_summary(
    freeciv_state: FreeCivState, player_id: int
) -> List[Dict[str, Any]]:
    """Format cities visible to player.

    Args:
        freeciv_state: The FreeCiv game state
        player_id: Player ID for visibility filtering

    Returns:
        List of formatted city information
    """
    visible_coords = freeciv_state._get_cached_visibility(player_id)
    formatted_cities = []

    for city in freeciv_state.cities.values():
        # Include own cities and visible enemy cities
        if city.owner == player_id or city.position in visible_coords:
            if city.owner == player_id:
                # Full details for own cities
                city_info = {
                    "id": city.city_id,
                    "name": city.name,
                    "position": {"x": city.position[0], "y": city.position[1]},
                    "owner": city.owner,
                    "population": city.population,
                    "production": city.production,
                    "specialists": city.specialists,
                    "buildings": city.buildings,
                    "food_storage": city.food_storage,
                    "shield_storage": city.shield_storage,
                    "celebrating": city.celebrating,
                    "disorder": city.disorder,
                    "under_siege": city.under_siege,
                    "available_actions": len(city.available_actions),
                }
            else:
                # Limited details for enemy cities
                city_info = {
                    "id": city.city_id,
                    "name": city.name,
                    "position": {"x": city.position[0], "y": city.position[1]},
                    "owner": city.owner,
                    "population": city.population,
                    "enemy_city": True,
                }
            formatted_cities.append(city_info)

    return formatted_cities


def format_state(freeciv_state: FreeCivState, player_id: int = 1) -> str:
    """Convert FreeCiv state to structured JSON string.

    Args:
        freeciv_state: The FreeCiv game state to format
        player_id: Player ID for perspective-specific formatting

    Returns:
        JSON string representation of the game state
    """
    player = freeciv_state.players.get(player_id)

    formatted_state = {
        "game_info": {
            "turn": freeciv_state.turn,
            "phase": freeciv_state.phase,
            "current_player": freeciv_state.current_player(),
            "is_terminal": freeciv_state.is_terminal(),
            "scores": freeciv_state._scores,
        },
        "map_info": {
            "width": freeciv_state.map.width,
            "height": freeciv_state.map.height,
            "grid": _format_map_grid(freeciv_state, player_id),
        },
        "player_info": {
            "id": player_id,
            "name": player.name if player else f"Player {player_id}",
            "nation": player.nation if player else "Unknown",
            "score": freeciv_state._scores.get(str(player_id), 0),
            "gold": player.gold if player else 0,
            "science": player.science if player else 0,
            "research_target": player.research_target if player else None,
            "research_progress": player.research_progress if player else 0,
            "government": player.government if player else None,
            "techs_count": len(player.techs) if player else 0,
            "diplomatic_relations": player.diplomatic_relations if player else {},
        },
        "units": _format_units_summary(freeciv_state, player_id),
        "cities": _format_cities_summary(freeciv_state, player_id),
        "legal_actions_count": len(freeciv_state.get_legal_actions(player_id)),
    }

    return json.dumps(formatted_state, indent=2)


def convert_state(state_str: str, detailed: bool = True) -> str:
    """Convert state string to more readable format.

    Args:
        state_str: String representation of FreeCiv state
        detailed: Whether to include detailed information

    Returns:
        Formatted state string
    """
    try:
        state_data = json.loads(state_str)
    except json.JSONDecodeError:
        # If not JSON, return as-is
        return state_str

    if not detailed:
        # Create simplified summary
        game_info = state_data.get("game_info", {})
        player_info = state_data.get("player_info", {})
        units = state_data.get("units", [])
        cities = state_data.get("cities", [])

        summary = f"""FreeCiv Game - Turn {game_info.get('turn', '?')}
Phase: {game_info.get('phase', 'unknown')}
Player: {player_info.get('name', 'Unknown')} ({player_info.get('nation', 'Unknown')})
Score: {player_info.get('score', 0)} | Gold: {player_info.get('gold', 0)}
Units: {len(units)} | Cities: {len(cities)}
Legal Actions: {state_data.get('legal_actions_count', 0)}
Research: {player_info.get('research_target', 'None')} ({player_info.get('research_progress', 0)}%)"""

        return summary

    return state_str


def format_action_summary(freeciv_state: FreeCivState, player_id: int) -> str:
    """Create human-readable summary of available actions.

    Args:
        freeciv_state: The FreeCiv game state
        player_id: Player ID to get actions for

    Returns:
        Formatted string describing available actions
    """
    legal_actions = freeciv_state.get_legal_actions(player_id)

    if not legal_actions:
        return "No legal actions available."

    # Group actions by type
    action_groups = {}
    for action in legal_actions:
        action_type = action.action_type
        if action_type not in action_groups:
            action_groups[action_type] = []
        action_groups[action_type].append(action)

    summary_lines = [f"Available Actions ({len(legal_actions)} total):"]

    for action_type, actions in action_groups.items():
        count = len(actions)
        examples = []
        for action in actions[:3]:  # Show up to 3 examples
            if action.source == "unit" and action.actor_id in freeciv_state.units:
                unit = freeciv_state.units[action.actor_id]
                examples.append(f"{unit.kind}({action.actor_id})")
            elif action.source == "city" and action.actor_id in freeciv_state.cities:
                city = freeciv_state.cities[action.actor_id]
                examples.append(f"{city.name}({action.actor_id})")
            else:
                examples.append(f"actor({action.actor_id})")

        example_str = ", ".join(examples)
        if count > 3:
            example_str += f", +{count-3} more"

        summary_lines.append(f"  {action_type}: {count} ({example_str})")

    return "\n".join(summary_lines)
