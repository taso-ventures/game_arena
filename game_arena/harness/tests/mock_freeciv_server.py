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

"""Mock FreeCiv server for testing WebSocket client functionality."""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Set

import websockets

logger = logging.getLogger(__name__)


class MockFreeCivServer:
    """Mock FreeCiv server that simulates FreeCiv3D proxy behavior for testing."""

    def __init__(self, host: str = "localhost", port: int = 8002):
        """Initialize mock server.

        Args:
            host: Server host address
            port: Server port number
        """
        self.host = host
        self.port = port
        self.server = None
        self.connected_clients: Set[Any] = set()
        self.agent_sessions: Dict[str, Dict[str, Any]] = {}
        self.turn_counter = 1
        self.game_state = self._get_default_game_state()
        self.message_delay = 0.01  # Configurable delay for testing
        self.should_fail_next = False  # For failure injection
        self.recorded_messages: List[Dict[str, Any]] = []  # For test verification

    def _get_default_game_state(self) -> Dict[str, Any]:
        """Get a default game state for testing.

        Returns protocol-compliant state_update message matching FreeCiv3D
        format. Players, units, cities are DICTS keyed by ID (Option A).

        Field type coercion:
        - nation: sent as string (coerced from int in CivCom)
        - activity: sent as None for idle (0), string for others
        """
        return {
            "type": "state_update",
            "timestamp": time.time(),
            "turn": self.turn_counter,
            "phase": "movement",
            "player_id": 0,  # Player perspective (real server includes this)
            "format": "full",
            "cached": False,
            # Game metadata
            "game": {
                "ruleset": "classic",
                "difficulty": "easy",
                "victory_conditions": ["conquest", "space_race"],
            },
            # Map information
            "map": {
                "width": 80,
                "height": 50,
                "topology": "wrapx",
            },
            # Players as DICT keyed by player_id (actual server format)
            "players": {
                "0": {
                    "playerno": 0,
                    "id": 0,
                    "name": "TestPlayer",
                    "nation": "18",  # String (coerced from int nation ID)
                    "is_alive": True,
                    "is_ai": False,
                    "gold": 50,
                    "science": 10,
                    "government": "despotism",
                    "researching": "alphabet",
                },
                "1": {
                    "playerno": 1,
                    "id": 1,
                    "name": "AI Player",
                    "nation": "408",  # String (coerced from int nation ID)
                    "is_alive": True,
                    "is_ai": True,
                    "gold": 40,
                    "science": 8,
                    "government": "despotism",
                    "researching": "pottery",
                },
            },
            # Units as DICT keyed by unit_id (actual server format)
            "units": {
                "101": {
                    "id": 101,
                    "unit_id": 101,  # Real server includes both id and unit_id
                    "type": "settlers",
                    "owner": 0,
                    "x": 1,
                    "y": 1,
                    "movesleft": 3,
                    "hp": 20,
                    "veteran_level": 0,
                    "activity": None,  # None = idle (coerced from int 0)
                },
                "102": {
                    "id": 102,
                    "unit_id": 102,  # Real server includes both id and unit_id
                    "type": "warrior",
                    "owner": 1,
                    "x": 10,
                    "y": 10,
                    "movesleft": 1,
                    "hp": 10,
                    "veteran_level": 0,
                    "activity": "2",  # String (coerced from int activity enum)
                },
            },
            # Cities as DICT keyed by city_id (actual server format)
            "cities": {
                "301": {
                    "id": 301,
                    "city_id": 301,  # Real server includes both id and city_id
                    "name": "Athens",
                    "owner": 0,
                    "x": 5,
                    "y": 5,
                    "size": 3,
                    "production": "warrior",
                    "food_stock": 10,
                    "shield_stock": 5,
                },
            },
        }

    async def start(self) -> None:
        """Start the mock server."""
        self.server = await websockets.serve(
            self.handle_connection, self.host, self.port
        )
        logger.info(f"Mock FreeCiv server started on {self.host}:{self.port}")

    async def stop(self) -> None:
        """Stop the mock server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("Mock FreeCiv server stopped")

    async def handle_connection(self, websocket: Any) -> None:
        """Handle new WebSocket connection.

        Args:
            websocket: WebSocket connection
        """
        self.connected_clients.add(websocket)
        logger.info(f"New client connected: {websocket.remote_address}")

        try:
            # Send welcome message
            welcome = {
                "type": "welcome",
                "handler_id": str(uuid.uuid4()),
                "message": "Mock FreeCiv server ready",
            }
            await websocket.send(json.dumps(welcome))

            async for message in websocket:
                await self._handle_message(websocket, message)

        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"Error handling connection: {e}")
        finally:
            self.connected_clients.discard(websocket)

    async def _handle_message(self, websocket: Any, message: str) -> None:
        """Handle incoming message from client.

        Args:
            websocket: WebSocket connection
            message: Raw message string
        """
        await asyncio.sleep(self.message_delay)  # Simulate network delay

        if self.should_fail_next:
            self.should_fail_next = False
            await websocket.close(code=1011, reason="Simulated server error")
            return

        try:
            data = json.loads(message)
            self.recorded_messages.append(data)
            msg_type = data.get("type", "")

            if msg_type == "llm_connect":
                await self._handle_llm_connect(websocket, data)
            elif msg_type == "state_query":
                await self._handle_state_query(websocket, data)
            elif msg_type == "action":
                await self._handle_action(websocket, data)
            elif msg_type == "ping":
                await self._handle_ping(websocket, data)
            else:
                await self._handle_unknown(websocket, data)

        except json.JSONDecodeError:
            error_response = {
                "type": "error",
                "code": "E001",
                "message": "Invalid JSON format",
            }
            await websocket.send(json.dumps(error_response))

    async def _handle_llm_connect(self, websocket: Any, data: Dict[str, Any]) -> None:
        """Handle LLM agent connection request."""
        agent_id = data.get("agent_id", "unknown")
        game_id = data.get("game_id", "default")

        # Store session info
        self.agent_sessions[agent_id] = {
            "websocket": websocket,
            "game_id": game_id,
            "player_id": 1,
            "connected_at": time.time(),
        }

        # Match the expected authentication response structure from FreeCiv3D
        response = {
            "type": "llm_connect",
            "data": {
                "type": "auth_success",
                "success": True,
                "player_id": 1,
                "game_id": game_id,
                "civserver_port": 5556,
                "capabilities": ["state_query", "action", "ping"],
            }
        }
        await websocket.send(json.dumps(response))

    async def _handle_state_query(self, websocket: Any, data: Dict[str, Any]) -> None:
        """Handle state query request."""
        format_type = data.get("format", "llm_optimized")

        # Update turn counter and timestamp for dynamic testing
        self.game_state["turn"] = self.turn_counter
        self.game_state["timestamp"] = time.time()

        response = self.game_state.copy()
        if format_type == "minimal":
            # Return minimal state for specific tests (still protocol-compliant)
            response = {
                "type": "state_update",
                "turn": self.turn_counter,
                "phase": "movement",
                "format": "minimal",
                "cached": False,
                "timestamp": time.time(),
                "game": {},
                "map": {},
                "players": {},  # Dicts (Option A format)
                "units": {},
                "cities": {},
            }

        await websocket.send(json.dumps(response))

    async def _handle_action(self, websocket: Any, data: Dict[str, Any]) -> None:
        """Handle action submission."""
        action_data = data.get("data", {})
        # Check for nested data structure (FreeCiv packet format)
        if "data" in action_data:
            action_type = action_data["data"].get("action_type", "unknown")
        else:
            action_type = action_data.get("action_type", "unknown")

        # Simulate action processing
        success = True
        result_data = {"action_processed": True}

        # Simulate some actions failing for testing
        if action_type == "invalid_action":
            success = False
            result_data = {"error": "Invalid action type"}

        response = {
            "type": "action_result",
            "success": success,
            "data": result_data,
            "timestamp": int(time.time()),
        }
        await websocket.send(json.dumps(response))

        # Advance turn after successful action
        if success:
            self.turn_counter += 1

    async def _handle_ping(self, websocket: Any, data: Dict[str, Any]) -> None:
        """Handle ping/heartbeat request."""
        response = {
            "type": "pong",
            "timestamp": int(time.time()),
            "echo": data.get("echo", ""),
        }
        await websocket.send(json.dumps(response))

    async def _handle_unknown(self, websocket: Any, data: Dict[str, Any]) -> None:
        """Handle unknown message type."""
        response = {
            "type": "error",
            "code": "E002",
            "message": f"Unknown message type: {data.get('type', 'undefined')}",
        }
        await websocket.send(json.dumps(response))

    async def simulate_disconnect(self, agent_id: str) -> None:
        """Simulate disconnection for specific agent (for testing)."""
        if agent_id in self.agent_sessions:
            websocket = self.agent_sessions[agent_id]["websocket"]
            await websocket.close(code=1000, reason="Simulated disconnect")
            del self.agent_sessions[agent_id]

    async def send_to_client(self, agent_id: str, message: str) -> None:
        """Send a message to a specific connected client.

        Args:
            agent_id: Agent identifier
            message: Message to send (JSON string)
        """
        if agent_id in self.agent_sessions:
            websocket = self.agent_sessions[agent_id]["websocket"]
            try:
                await websocket.send(message)
                logger.debug(f"Sent message to agent {agent_id}: {message[:100]}")
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"Failed to send to {agent_id}: connection closed")
                del self.agent_sessions[agent_id]
        else:
            logger.warning(f"Agent {agent_id} not found in active sessions")

    async def broadcast_turn_notification(self) -> None:
        """Broadcast turn change to all connected agents."""
        self.turn_counter += 1
        notification = {
            "type": "turn_notification",
            "data": {
                "turn": self.turn_counter,
                "current_player": 1,
                "phase": "movement",
            },
            "timestamp": int(time.time()),
        }

        message = json.dumps(notification)
        disconnected = []

        for websocket in self.connected_clients:
            try:
                await websocket.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.append(websocket)

        # Clean up disconnected clients
        for websocket in disconnected:
            self.connected_clients.discard(websocket)

    def inject_failure(self) -> None:
        """Inject failure on next message (for testing resilience)."""
        self.should_fail_next = True

    def set_message_delay(self, delay: float) -> None:
        """Set artificial message delay for testing."""
        self.message_delay = delay

    def get_recorded_messages(self) -> List[Dict[str, Any]]:
        """Get all messages received by the server (for test verification)."""
        return self.recorded_messages.copy()

    def clear_recorded_messages(self) -> None:
        """Clear recorded messages."""
        self.recorded_messages.clear()

    def get_connected_agent_count(self) -> int:
        """Get number of connected agents."""
        return len(self.agent_sessions)
