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

"""Test helpers and utilities for FreeCiv integration tests."""

import json
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

from game_arena.harness.freeciv_proxy_client import FreeCivProxyClient
from game_arena.harness.freeciv_state import FreeCivAction, FreeCivState


class MockFreeCivProxyClient:
  """Mock FreeCiv proxy client that simulates real server responses."""

  def __init__(self, use_recorded_responses: bool = True):
    """Initialize mock client.

    Args:
      use_recorded_responses: Whether to use pre-recorded responses
    """
    self.use_recorded_responses = use_recorded_responses
    self.recorded_responses = self._load_recorded_responses()
    self.connection_status = False
    self.call_history = []

  def _load_recorded_responses(self) -> Dict[str, Any]:
    """Load pre-recorded server responses for testing."""
    return {
      "get_game_state": {
        "turn": 5,
        "phase": "move",
        "current_player": 1,
        "players": {
          "1": {
            "id": 1,
            "name": "Romans",
            "score": 150,
            "gold": 50,
            "science": 25,
            "culture": 10,
            "cities": 1,
            "units": 2,
            "techs": ["Pottery", "Animal Husbandry"]
          },
          "2": {
            "id": 2,
            "name": "Greeks",
            "score": 140,
            "gold": 45,
            "science": 20,
            "culture": 8,
            "cities": 1,
            "units": 2,
            "techs": ["Pottery", "Mysticism"]
          }
        },
        "units": [
          {
            "id": 101,
            "type": "Warrior",
            "x": 10,
            "y": 14,
            "hp": 10,
            "max_hp": 10,
            "mp": 1,
            "max_mp": 1,
            "owner": 1,
            "veteran": False,
            "fortified": False,
            "activity": "idle"
          },
          {
            "id": 102,
            "type": "Settler",
            "x": 12,
            "y": 16,
            "hp": 20,
            "max_hp": 20,
            "mp": 1,
            "max_mp": 1,
            "owner": 1,
            "veteran": False,
            "fortified": False,
            "activity": "idle"
          }
        ],
        "cities": [
          {
            "id": 201,
            "name": "Rome",
            "x": 10,
            "y": 15,
            "size": 3,
            "owner": 1,
            "production": {
              "name": "Warrior",
              "progress": 5,
              "total": 10
            },
            "food_surplus": 2,
            "trade": 3,
            "corruption": 0,
            "buildings": ["Palace"]
          }
        ],
        "map": {
          "width": 80,
          "height": 50,
          "tiles": [
            {"x": 10, "y": 14, "terrain": "grassland", "special": None},
            {"x": 11, "y": 14, "terrain": "forest", "special": None},
            {"x": 10, "y": 15, "terrain": "grassland", "special": None},
            {"x": 12, "y": 16, "terrain": "plains", "special": None}
          ]
        },
        "research": {
          "current": "Bronze Working",
          "progress": 8,
          "cost": 12
        }
      },
      "send_action": {
        "success": True,
        "action_id": 1,
        "message": "Action executed successfully",
        "new_state": {
          "turn": 5,
          "phase": "move",
          "units_moved": 1
        }
      },
      "get_legal_actions": [
        {
          "action_type": "unit_move",
          "actor_id": 101,
          "target": {"x": 9, "y": 14},
          "valid": True
        },
        {
          "action_type": "unit_move",
          "actor_id": 101,
          "target": {"x": 11, "y": 14},
          "valid": True
        },
        {
          "action_type": "unit_fortify",
          "actor_id": 101,
          "target": {},
          "valid": True
        },
        {
          "action_type": "unit_move",
          "actor_id": 102,
          "target": {"x": 13, "y": 16},
          "valid": True
        },
        {
          "action_type": "city_production",
          "actor_id": 201,
          "target": {"value": "Warrior"},
          "valid": True
        }
      ]
    }

  async def connect(self) -> bool:
    """Mock connection to FreeCiv server."""
    self.call_history.append({"method": "connect", "timestamp": time.time()})
    self.connection_status = True
    return True

  async def disconnect(self) -> None:
    """Mock disconnection from FreeCiv server."""
    self.call_history.append({"method": "disconnect", "timestamp": time.time()})
    self.connection_status = False

  async def get_game_state(self) -> Dict[str, Any]:
    """Get mock game state."""
    self.call_history.append({"method": "get_game_state", "timestamp": time.time()})

    if not self.connection_status:
      raise ConnectionError("Not connected to FreeCiv server")

    return self.recorded_responses["get_game_state"].copy()

  async def send_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
    """Send mock action to server."""
    self.call_history.append({
      "method": "send_action",
      "action": action,
      "timestamp": time.time()
    })

    if not self.connection_status:
      raise ConnectionError("Not connected to FreeCiv server")

    # Simulate action result
    result = self.recorded_responses["send_action"].copy()
    result["action_sent"] = action
    return result

  async def get_legal_actions(self, player_id: int = 1) -> List[Dict[str, Any]]:
    """Get mock legal actions."""
    self.call_history.append({
      "method": "get_legal_actions",
      "player_id": player_id,
      "timestamp": time.time()
    })

    if not self.connection_status:
      raise ConnectionError("Not connected to FreeCiv server")

    return self.recorded_responses["get_legal_actions"].copy()

  def is_connected(self) -> bool:
    """Check if mock client is connected."""
    return self.connection_status

  def get_call_history(self) -> List[Dict[str, Any]]:
    """Get history of all method calls."""
    return self.call_history.copy()

  def reset(self) -> None:
    """Reset mock client state."""
    self.call_history.clear()
    self.connection_status = False


class MockModelWithPredictableResponses:
  """Mock model that returns predictable FreeCiv actions for testing."""

  def __init__(self, model_name: str = "mock-freeciv-model"):
    """Initialize mock model.

    Args:
      model_name: Name of the mock model
    """
    self.model_name = model_name
    self.call_count = 0
    self.predefined_responses = [
      "unit_move_warrior(101)_to(11,14)",
      "unit_fortify_warrior(101)",
      "city_production_rome(201)_target(warrior)",
      "unit_move_settler(102)_to(13,16)",
      "unit_explore_warrior(101)"
    ]

  async def generate_with_text_input(self, prompt: str, **kwargs) -> str:
    """Generate predictable response for testing with correct interface."""
    response = self.predefined_responses[self.call_count % len(self.predefined_responses)]
    self.call_count += 1
    return response

  async def generate_response(self, prompt: str, **kwargs) -> str:
    """Generate predictable response for testing (alternate interface)."""
    return await self.generate_with_text_input(prompt, **kwargs)

  def reset(self) -> None:
    """Reset mock model state."""
    self.call_count = 0


class FreeCivTestData:
  """Container for test data and fixtures."""

  @staticmethod
  def create_sample_observation(player_id: int = 1, turn: int = 5) -> Dict[str, Any]:
    """Create sample observation for testing."""
    return {
      "playerID": player_id,
      "turn": turn,
      "phase": "move",
      "serializedGameAndState": "mock_encoded_state",
      "legalActions": [0, 1, 2, 3, 4],
      "players": {
        str(player_id): {
          "id": player_id,
          "name": "Romans" if player_id == 1 else "Greeks",
          "score": 150,
          "gold": 50
        }
      },
      "units": [
        {
          "id": 101,
          "type": "Warrior",
          "x": 10,
          "y": 14,
          "owner": player_id
        }
      ],
      "cities": [
        {
          "id": 201,
          "name": "Rome" if player_id == 1 else "Athens",
          "x": 10,
          "y": 15,
          "owner": player_id
        }
      ]
    }

  @staticmethod
  def create_sample_freeciv_state(player_id: int = 1, turn: int = 5) -> FreeCivState:
    """Create sample FreeCivState for testing."""
    state_data = {
      "game": {
        "turn": turn,
        "phase": "move",
        "current_player": player_id,
        "year": 4000,
        "turn_timeout": 0
      },
      "players": [
        {
          "id": player_id,
          "name": "Romans" if player_id == 1 else "Greeks",
          "score": 150,
          "gold": 50,
          "science": 25,
          "is_alive": True,
          "government": "Despotism"
        }
      ],
      "units": [
        {
          "id": 101,
          "type": "Warrior",
          "x": 10,
          "y": 14,
          "hp": 10,
          "max_hp": 10,
          "mp": 1,
          "max_mp": 1,
          "owner": player_id,
          "veteran": False,
          "activity": "idle"
        }
      ],
      "cities": [
        {
          "id": 201,
          "name": "Rome" if player_id == 1 else "Athens",
          "x": 10,
          "y": 15,
          "population": 3,
          "owner": player_id,
          "food_surplus": 2,
          "shield_surplus": 1,
          "trade": 3
        }
      ],
      "map": {
        "width": 80,
        "height": 50,
        "topology": 0,
        "tiles": [
          {
            "x": 10,
            "y": 14,
            "terrain": "grassland",
            "special": None,
            "continent": 1
          },
          {
            "x": 11,
            "y": 14,
            "terrain": "forest",
            "special": None,
            "continent": 1
          }
        ]
      }
    }
    return FreeCivState(state_data)

  @staticmethod
  def create_sample_freeciv_actions() -> List[FreeCivAction]:
    """Create sample FreeCiv actions for testing."""
    return [
      FreeCivAction(
        action_type="unit_move",
        actor_id=101,
        target={"x": 11, "y": 14},
        parameters={},
        source="unit"
      ),
      FreeCivAction(
        action_type="unit_fortify",
        actor_id=101,
        target={},
        parameters={},
        source="unit"
      ),
      FreeCivAction(
        action_type="city_production",
        actor_id=201,
        target={"value": "warrior"},
        parameters={},
        source="city"
      )
    ]


class AsyncTestHelpers:
  """Helpers for async testing operations."""

  @staticmethod
  def run_async_test(coro):
    """Run async test function synchronously."""
    import asyncio
    return asyncio.run(coro)

  @staticmethod
  async def simulate_game_turn(mock_client: MockFreeCivProxyClient,
                               actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Simulate a full game turn with multiple actions."""
    results = []

    # Get initial state
    initial_state = await mock_client.get_game_state()

    # Execute each action
    for action in actions:
      result = await mock_client.send_action(action)
      results.append(result)

    return results

  @staticmethod
  def assert_valid_freeciv_action(action: FreeCivAction) -> None:
    """Assert that a FreeCivAction is valid."""
    assert hasattr(action, 'action_type'), "Action must have action_type"
    assert hasattr(action, 'actor_id'), "Action must have actor_id"
    assert hasattr(action, 'target'), "Action must have target"
    assert hasattr(action, 'parameters'), "Action must have parameters"
    assert hasattr(action, 'source'), "Action must have source"

    assert isinstance(action.action_type, str), "action_type must be string"
    assert isinstance(action.actor_id, int), "actor_id must be integer"
    assert isinstance(action.target, dict), "target must be dictionary"
    assert isinstance(action.parameters, dict), "parameters must be dictionary"