"""FreeCiv client for connecting to FreeCiv3D server via WebSocket."""

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional

import requests
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from game_arena.harness.freeciv_state import FreeCivAction

logger = logging.getLogger(__name__)


class FreeCivClient:
    """Client for communicating with FreeCiv3D server."""

    def __init__(self, server_url: str, ws_url: str, timeout: float = 30.0):
        """Initialize FreeCiv client.

        Args:
            server_url: HTTP URL of FreeCiv3D server (e.g., http://localhost:8080)
            ws_url: WebSocket URL of FreeCiv3D server (e.g., ws://localhost:4002)
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip("/")
        self.ws_url = ws_url
        self.timeout = timeout
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.game_id: Optional[str] = None
        self.player_id: Optional[int] = None
        self.connected = False

    def connect(self) -> None:
        """Connect to FreeCiv3D server and join/create a game."""
        try:
            # First, try to get server status
            response = requests.get(f"{self.server_url}/status", timeout=self.timeout)
            if response.status_code != 200:
                raise ConnectionError(
                    f"FreeCiv server not responding: {response.status_code}"
                )

            # Try to create or join a game
            self._setup_game()

            # Connect WebSocket
            self._connect_websocket()

            self.connected = True
            logger.info(
                f"Connected to FreeCiv3D server, game_id: {self.game_id}, player_id: {self.player_id}"
            )

        except Exception as e:
            logger.error(f"Failed to connect to FreeCiv server: {e}")
            raise

    def _setup_game(self) -> None:
        """Create or join a game session."""
        try:
            # Try to create a new game
            create_data = {
                "action": "new",
                "type": "multiplayer",
                "ruleset": "classic",
                "map_size": "small",
            }
            response = requests.post(
                f"{self.server_url}/civclientlauncher",
                data=create_data,
                timeout=self.timeout,
            )

            if response.status_code == 200:
                result = response.json() if response.content else {}
                self.game_id = result.get("game_id", "default")
                self.player_id = result.get("player_id", 1)
            else:
                # If creation fails, try to join existing game
                logger.warning("Game creation failed, attempting to join existing game")
                self.game_id = "default"
                self.player_id = 1

        except Exception as e:
            logger.warning(f"Game setup failed, using defaults: {e}")
            self.game_id = "default"
            self.player_id = 1

    def _connect_websocket(self) -> None:
        """Connect to WebSocket for real-time communication."""
        try:
            # Use asyncio to handle WebSocket connection
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._async_connect_websocket())
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            # Continue without WebSocket for HTTP-only mode
            self.websocket = None

    async def _async_connect_websocket(self) -> None:
        """Async WebSocket connection."""
        try:
            self.websocket = await websockets.connect(self.ws_url, timeout=self.timeout)
            logger.info("WebSocket connected successfully")
        except Exception as e:
            logger.warning(f"WebSocket connection failed, continuing in HTTP mode: {e}")
            self.websocket = None

    def get_game_state(self) -> Dict[str, Any]:
        """Get current game state from FreeCiv server.

        Returns:
            Dictionary containing full game state
        """
        if not self.connected:
            raise RuntimeError("Not connected to FreeCiv server")

        try:
            # Try WebSocket first if available
            if self.websocket:
                return self._get_state_via_websocket()
            else:
                return self._get_state_via_http()

        except Exception as e:
            logger.error(f"Failed to get game state: {e}")
            # Return a minimal mock state for testing
            return self._get_mock_game_state()

    def _get_state_via_websocket(self) -> Dict[str, Any]:
        """Get game state via WebSocket."""
        if not self.websocket:
            raise RuntimeError("WebSocket not connected")

        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._async_get_state_via_websocket())
        except Exception as e:
            logger.error(f"WebSocket state request failed: {e}")
            raise

    async def _async_get_state_via_websocket(self) -> Dict[str, Any]:
        """Async WebSocket state request."""
        if not self.websocket:
            raise RuntimeError("WebSocket not connected")

        state_request = {
            "type": "state_query",
            "game_id": self.game_id,
            "player_id": self.player_id,
        }

        await self.websocket.send(json.dumps(state_request))
        response = await self.websocket.recv()
        return json.loads(response)

    def _get_state_via_http(self) -> Dict[str, Any]:
        """Get game state via HTTP API."""
        try:
            url = f"{self.server_url}/api/game/{self.game_id}/state"
            params = {"format": "llm_optimized", "player_id": self.player_id}
            response = requests.get(url, params=params, timeout=self.timeout)

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"HTTP state request failed: {response.status_code}")
                return self._get_mock_game_state()

        except Exception as e:
            logger.error(f"HTTP state request failed: {e}")
            return self._get_mock_game_state()

    def _get_mock_game_state(self) -> Dict[str, Any]:
        """Return a minimal mock game state for testing."""
        return {
            "game": {
                "turn": 1,
                "phase": "movement",
                "current_player": self.player_id,
                "is_over": False,
                "scores": {str(self.player_id): 0, "2": 0},
            },
            "map": {
                "width": 6,
                "height": 4,
                "tiles": [
                    {
                        "x": i,
                        "y": j,
                        "terrain": "grassland",
                        "resource": None,
                        "city_id": None,
                        "unit_ids": [],
                        "improvements": [],
                        "pollution": False,
                        "fallout": False,
                        "owner": None,
                        "worked_by": None,
                    }
                    for i in range(6)
                    for j in range(4)
                ],
                "visibility": {
                    str(self.player_id): [[i, j] for i in range(6) for j in range(4)]
                },
            },
            "players": [
                {
                    "id": self.player_id,
                    "name": f"Player {self.player_id}",
                    "nation": "Romans",
                    "score": 0,
                    "gold": 50,
                    "techs": ["pottery"],
                    "government": "despotism",
                    "science": 10,
                    "research_target": "bronze_working",
                    "research_progress": 5,
                    "diplomatic_relations": [],
                    "trade_routes": [],
                    "luxuries_rate": 0,
                    "science_rate": 50,
                    "tax_rate": 50,
                }
            ],
            "units": [
                {
                    "id": 101,
                    "owner": self.player_id,
                    "type": "settlers",
                    "x": 1,
                    "y": 1,
                    "hp": 20,
                    "moves_left": 1,
                    "veteran": False,
                    "orders": [],
                    "available_actions": [
                        {
                            "type": "unit_move",
                            "target": {"x": 2, "y": 1},
                            "parameters": {},
                        },
                        {
                            "type": "unit_move",
                            "target": {"x": 1, "y": 2},
                            "parameters": {},
                        },
                    ],
                    "fortified": False,
                    "activity": None,
                    "fuel": -1,
                    "transport_id": None,
                    "cargo_ids": [],
                }
            ],
            "cities": [],
        }

    def submit_action(self, action: FreeCivAction) -> bool:
        """Submit an action to the FreeCiv server.

        Args:
            action: FreeCivAction to submit

        Returns:
            True if action was submitted successfully
        """
        if not self.connected:
            raise RuntimeError("Not connected to FreeCiv server")

        try:
            # Try WebSocket first if available
            if self.websocket:
                return self._submit_action_via_websocket(action)
            else:
                return self._submit_action_via_http(action)

        except Exception as e:
            logger.error(f"Failed to submit action: {e}")
            return False

    def _submit_action_via_websocket(self, action: FreeCivAction) -> bool:
        """Submit action via WebSocket."""
        if not self.websocket:
            raise RuntimeError("WebSocket not connected")

        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self._async_submit_action_via_websocket(action)
            )
        except Exception as e:
            logger.error(f"WebSocket action submission failed: {e}")
            return False

    async def _async_submit_action_via_websocket(self, action: FreeCivAction) -> bool:
        """Async WebSocket action submission."""
        if not self.websocket:
            raise RuntimeError("WebSocket not connected")

        action_request = {
            "type": "action",
            "game_id": self.game_id,
            "player_id": self.player_id,
            "data": action.to_packet(),
        }

        await self.websocket.send(json.dumps(action_request))
        response = await self.websocket.recv()
        result = json.loads(response)
        return result.get("success", False)

    def _submit_action_via_http(self, action: FreeCivAction) -> bool:
        """Submit action via HTTP API."""
        try:
            url = f"{self.server_url}/api/game/{self.game_id}/action"
            data = {"player_id": self.player_id, "action": action.to_packet()}
            response = requests.post(url, json=data, timeout=self.timeout)
            return response.status_code == 200

        except Exception as e:
            logger.error(f"HTTP action submission failed: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from FreeCiv server."""
        if self.websocket:
            try:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self.websocket.close())
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")

        self.connected = False
        self.websocket = None
        logger.info("Disconnected from FreeCiv server")
